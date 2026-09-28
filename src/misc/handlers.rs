//! Progress, status and interruption handlers (`igraph_progress.h`,
//! `igraph_statusbar.h`, `igraph_interrupt.h`).
//!
//! igraph keeps one handler of each kind *per thread*. The C handlers take
//! no user data, so the Rust closures live in thread-local storage of this
//! module and a C trampoline dispatches to them. Installing a handler
//! returns a guard, and the handler stays installed while its guard lives.
//! The guards of each kind form a per-thread stack: the active handler is
//! the one of the most recently created guard that is still alive, and when
//! no guard is left, whatever handler was active before the first one is
//! reinstalled. Guards may be dropped in any order: dropping a guard always
//! removes exactly its own handler (reinstalling the next one only if it was
//! the active one).
//!
//! The handler closures may call igraph themselves: they run in their own
//! level of igraph's cleanup ("finally") stack, so even a failing call made
//! by a handler does not disturb the computation that invoked it. They must
//! not start ARPACK-based computations while an ARPACK-based computation is
//! running on the same thread (see [`set_interruption_handler`]).

use super::lossy;
use crate::{
    error::{Error, Result, catch_panic, catch_panic_or, ensure_init, resume_panic},
    ffi::*,
    igraph_call,
};
use std::{
    cell::RefCell,
    ffi::{CString, c_char, c_void},
    marker::PhantomData,
    ops::ControlFlow,
    thread::LocalKey,
};

type ProgressDyn = dyn FnMut(&str, f64) -> ControlFlow<()>;
type StatusDyn = dyn FnMut(&str) -> ControlFlow<()>;
type InterruptDyn = dyn FnMut() -> bool;

/// A handler installed by a guard that is still alive.
struct Entry<F: ?Sized, C> {
    /// Identifies the guard.
    id: u64,
    /// The C handler to install while this entry is on top.
    c: C,
    /// The Rust closure of an entry that is not on top (the closure of the
    /// top entry lives in [`HandlerStack::active`]). `None` for C-only
    /// handlers, and while the closure is running.
    parked: Option<Box<F>>,
}

/// The per-thread stack of installed handlers of one kind.
struct HandlerStack<F: ?Sized, C> {
    /// The C handler that was active when the stack was empty.
    base: C,
    entries: Vec<Entry<F, C>>,
    /// The closure of the top entry, with that entry's id; taken out while
    /// it runs.
    active: Option<(u64, Box<F>)>,
    next_id: u64,
}

type Stack<F, C> = RefCell<HandlerStack<F, C>>;

thread_local! {
    static PROGRESS: Stack<ProgressDyn, igraph_progress_handler_t> =
        const { RefCell::new(HandlerStack { base: None, entries: Vec::new(), active: None, next_id: 0 }) };
    static STATUS: Stack<StatusDyn, igraph_status_handler_t> =
        const { RefCell::new(HandlerStack { base: None, entries: Vec::new(), active: None, next_id: 0 }) };
    static INTERRUPT: Stack<InterruptDyn, igraph_interruption_handler_t> =
        const { RefCell::new(HandlerStack { base: None, entries: Vec::new(), active: None, next_id: 0 }) };
}

/// Pushes a handler on the stack of `key`, installing `c` with `set` (which
/// returns the previous C handler). Returns the id of the new entry.
fn push<F: ?Sized + 'static, C: Copy + 'static>(
    key: &'static LocalKey<Stack<F, C>>,
    set: fn(C) -> C,
    c: C,
    rust: Option<Box<F>>,
) -> u64 {
    ensure_init();
    key.with(|stack| {
        let mut stack = stack.borrow_mut();
        let prev = set(c);
        if stack.entries.is_empty() {
            stack.base = prev;
            // Nothing can be active without an entry.
            debug_assert!(stack.active.is_none());
        } else if let Some((id, f)) = stack.active.take() {
            // Park the closure of the current top.
            if let Some(top) = stack.entries.last_mut()
                && top.id == id
            {
                top.parked = Some(f);
            }
        }
        let id = stack.next_id;
        stack.next_id += 1;
        stack.entries.push(Entry {
            id,
            c,
            parked: None,
        });
        stack.active = rust.map(|f| (id, f));
        id
    })
}

/// Removes the entry `id` from the stack of `key`; if it was on top,
/// reinstalls the handler of the new top (or the base handler).
fn remove<F: ?Sized + 'static, C: Copy + 'static>(
    key: &'static LocalKey<Stack<F, C>>,
    set: fn(C) -> C,
    id: u64,
) {
    // Closures are dropped after the borrow is released: they may own other
    // guards, whose drop borrows the stack again.
    let leftovers = key.try_with(|stack| {
        let mut stack = stack.borrow_mut();
        let Some(pos) = stack.entries.iter().position(|e| e.id == id) else {
            return (None, None);
        };
        let removed = stack.entries.remove(pos);
        if pos < stack.entries.len() {
            // Not the active handler: nothing else changes.
            return (removed.parked, None);
        }
        let active = stack.active.take().map(|(_, f)| f);
        let c = match stack.entries.last_mut() {
            Some(top) => {
                let next = top.parked.take().map(|f| (top.id, f));
                let c = top.c;
                stack.active = next;
                c
            }
            None => stack.base,
        };
        set(c);
        (removed.parked, active)
    });
    drop(leftovers);
}

/// Calls the active closure of `key` (or returns `default` if there is none).
///
/// The closure is taken out of the stack for the duration of the call, so
/// that a re-entrant call (e.g. the closure itself running an igraph
/// function that reports progress) finds no closure instead of a busy
/// `RefCell`. It is put back afterwards, even if it panics: as the active
/// closure if its entry is still on top, parked in its entry if other
/// handlers were installed meanwhile, or dropped if its guard was dropped.
fn call_with<F: ?Sized + 'static, C: 'static, R>(
    key: &'static LocalKey<Stack<F, C>>,
    default: R,
    call: impl FnOnce(&mut F) -> R,
) -> R {
    struct PutBack<F: ?Sized + 'static, C: 'static> {
        key: &'static LocalKey<Stack<F, C>>,
        id: u64,
        handler: Option<Box<F>>,
    }
    impl<F: ?Sized + 'static, C: 'static> Drop for PutBack<F, C> {
        fn drop(&mut self) {
            let Some(handler) = self.handler.take() else {
                return;
            };
            let id = self.id;
            let leftover = self
                .key
                .try_with(|stack| {
                    let mut stack = stack.borrow_mut();
                    let stack = &mut *stack;
                    let is_top = stack.entries.last().is_some_and(|e| e.id == id);
                    if is_top && stack.active.is_none() {
                        stack.active = Some((id, handler));
                        return None;
                    }
                    match stack.entries.iter_mut().find(|e| e.id == id) {
                        Some(entry) if !is_top && entry.parked.is_none() => {
                            entry.parked = Some(handler);
                            None
                        }
                        _ => Some(handler),
                    }
                })
                .ok()
                .flatten();
            drop(leftover);
        }
    }
    let Some((id, handler)) = key
        .try_with(|stack| stack.borrow_mut().active.take())
        .ok()
        .flatten()
    else {
        return default;
    };
    let mut guard = PutBack {
        key,
        id,
        handler: Some(handler),
    };
    match guard.handler.as_deref_mut() {
        Some(f) => call(f),
        None => default,
    }
}

/// Runs a handler closure in a fresh level of igraph's "finally" stack.
///
/// igraph calls the handlers from inside running computations, which keep
/// their temporary objects on the finally stack; igraph's error handler frees
/// the current level when a call fails. Without the new level, a failing
/// igraph call made by the handler (whose `Err` the handler may ignore)
/// would free the temporaries of the computation that is still running.
fn in_finally_level<R>(f: impl FnOnce() -> R) -> R {
    // SAFETY: plain bookkeeping on igraph's thread-local finally stack. The
    // closures passed here are wrapped in `catch_panic`/`catch_panic_or` and
    // never unwind, so the matching EXIT always runs.
    unsafe { IGRAPH_FINALLY_ENTER() };
    let res = f();
    unsafe { IGRAPH_FINALLY_EXIT() };
    res
}

fn flow_to_code(flow: ControlFlow<()>) -> igraph_error_t {
    match flow {
        ControlFlow::Continue(()) => igraph_error_type_t_IGRAPH_SUCCESS,
        ControlFlow::Break(()) => igraph_error_type_t_IGRAPH_INTERRUPTED,
    }
}

fn to_cstring(message: &str) -> Result<CString> {
    CString::new(message).map_err(|_| Error::invalid("message contains an interior NUL byte"))
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

unsafe extern "C" fn progress_trampoline(
    message: *const c_char,
    percent: igraph_real_t,
    _data: *mut c_void,
) -> igraph_error_t {
    in_finally_level(|| {
        catch_panic(|| {
            let message = lossy(message);
            flow_to_code(call_with(&PROGRESS, ControlFlow::Continue(()), |f| {
                f(&message, percent)
            }))
        })
    })
}

fn set_progress(c: igraph_progress_handler_t) -> igraph_progress_handler_t {
    unsafe { igraph_set_progress_handler(c) }
}

/// Keeps a progress handler installed; returned by [`set_progress_handler`]
/// and [`set_progress_handler_stderr`].
///
/// Dropping the guard uninstalls its handler: if it was the active one, the
/// handler of the most recent guard still alive (or, if none, the handler
/// that was active before the first guard) is reinstalled. Guards can be
/// dropped in any order, e.g. when kept in a struct or a `Vec`.
///
/// The guard is bound to the thread that created it (it is neither `Send`
/// nor `Sync`), like igraph's handlers.
#[must_use = "the handler is uninstalled as soon as the guard is dropped"]
pub struct ProgressHandlerGuard {
    id: u64,
    _not_send: PhantomData<*const ()>,
}

impl std::fmt::Debug for ProgressHandlerGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressHandlerGuard")
            .finish_non_exhaustive()
    }
}

impl Drop for ProgressHandlerGuard {
    fn drop(&mut self) {
        remove(&PROGRESS, set_progress, self.id);
    }
}

fn install_progress(
    c: igraph_progress_handler_t,
    rust: Option<Box<ProgressDyn>>,
) -> ProgressHandlerGuard {
    ProgressHandlerGuard {
        id: push(&PROGRESS, set_progress, c, rust),
        _not_send: PhantomData,
    }
}

/// Installs a Rust closure as the progress handler of the calling thread
/// until the returned guard is dropped.
///
/// igraph functions performing long computations (e.g. betweenness and
/// closeness centralities, some layouts, the fast-greedy community
/// detection) periodically call the handler with a short message describing
/// the algorithm and the percentage of the work done (e.g.
/// [`Graph::betweenness`](crate::Graph::betweenness) reports
/// `"Betweenness centrality: "`): the first call has
/// `0.0` and the last one `100.0` (unless an error occurs). Returning
/// [`ControlFlow::Break`] aborts the computation, which then fails with
/// [`ErrorKind::Interrupted`](crate::ErrorKind::Interrupted). A panic in the
/// closure also aborts the computation, and is resumed when the wrapper
/// returns.
///
/// Binds [`igraph_set_progress_handler`](https://igraph.org/c/html/latest/igraph-Advanced.html#igraph_set_progress_handler).
///
/// # Examples
///
/// ```
/// use igraph::misc;
/// use std::{cell::RefCell, ops::ControlFlow, rc::Rc};
///
/// let seen = Rc::new(RefCell::new(Vec::new()));
/// let sink = Rc::clone(&seen);
/// let guard = misc::set_progress_handler(move |msg, pct| {
///     sink.borrow_mut().push((msg.to_owned(), pct));
///     ControlFlow::Continue(())
/// });
/// misc::progress("Crunching numbers", 50.0)?;
/// drop(guard);
/// misc::progress("Nobody listens", 100.0)?; // no handler anymore
/// assert_eq!(*seen.borrow(), vec![("Crunching numbers".to_owned(), 50.0)]);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn set_progress_handler(
    handler: impl FnMut(&str, f64) -> ControlFlow<()> + 'static,
) -> ProgressHandlerGuard {
    install_progress(Some(progress_trampoline), Some(Box::new(handler)))
}

/// Installs igraph's predefined progress handler, which prints the message
/// and the percentage to standard error, until the guard is dropped.
///
/// Binds [`igraph_progress_handler_stderr`](https://igraph.org/c/html/latest/igraph-Advanced.html#igraph_progress_handler_stderr).
pub fn set_progress_handler_stderr() -> ProgressHandlerGuard {
    install_progress(Some(igraph_progress_handler_stderr), None)
}

/// Runs `body` with `handler` installed as the progress handler of the
/// calling thread, restoring the previous handler afterwards (even if `body`
/// panics). See [`set_progress_handler`].
///
/// # Examples
///
/// ```
/// use igraph::misc;
/// use std::{cell::Cell, ops::ControlFlow, rc::Rc};
///
/// let calls = Rc::new(Cell::new(0));
/// let counter = Rc::clone(&calls);
/// let stopped = misc::with_progress_handler(
///     move |_, _| {
///         counter.set(counter.get() + 1);
///         ControlFlow::Break(()) // please stop
///     },
///     || misc::progress("step", 10.0),
/// );
/// assert_eq!(stopped.unwrap_err().kind(), igraph::ErrorKind::Interrupted);
/// assert_eq!(calls.get(), 1);
/// ```
pub fn with_progress_handler<R>(
    handler: impl FnMut(&str, f64) -> ControlFlow<()> + 'static,
    body: impl FnOnce() -> R,
) -> R {
    let _guard = set_progress_handler(handler);
    body()
}

/// Reports progress to the installed progress handler of the calling thread,
/// exactly as igraph's own functions do; a no-op when no handler is
/// installed. Useful to let long running Rust code built on this crate share
/// the same progress reporting channel.
///
/// Binds [`igraph_progress`](https://igraph.org/c/html/latest/igraph-Advanced.html#igraph_progress).
///
/// # Errors
/// [`ErrorKind::Interrupted`](crate::ErrorKind::Interrupted) if the handler
/// asked to stop; [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
/// if `message` contains a NUL byte.
pub fn progress(message: &str, percent: f64) -> Result<()> {
    let message = to_cstring(message)?;
    igraph_call!(igraph_progress(
        message.as_ptr(),
        percent,
        std::ptr::null_mut()
    ))
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

unsafe extern "C" fn status_trampoline(
    message: *const c_char,
    _data: *mut c_void,
) -> igraph_error_t {
    in_finally_level(|| {
        catch_panic(|| {
            let message = lossy(message);
            flow_to_code(call_with(&STATUS, ControlFlow::Continue(()), |f| {
                f(&message)
            }))
        })
    })
}

fn set_status(c: igraph_status_handler_t) -> igraph_status_handler_t {
    unsafe { igraph_set_status_handler(c) }
}

/// Keeps a status handler installed; returned by [`set_status_handler`]
/// and [`set_status_handler_stderr`].
///
/// Like [`ProgressHandlerGuard`], guards can be dropped in any order, and
/// each is bound to the creating thread.
#[must_use = "the handler is uninstalled as soon as the guard is dropped"]
pub struct StatusHandlerGuard {
    id: u64,
    _not_send: PhantomData<*const ()>,
}

impl std::fmt::Debug for StatusHandlerGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatusHandlerGuard").finish_non_exhaustive()
    }
}

impl Drop for StatusHandlerGuard {
    fn drop(&mut self) {
        remove(&STATUS, set_status, self.id);
    }
}

fn install_status(c: igraph_status_handler_t, rust: Option<Box<StatusDyn>>) -> StatusHandlerGuard {
    StatusHandlerGuard {
        id: push(&STATUS, set_status, c, rust),
        _not_send: PhantomData,
    }
}

/// Installs a Rust closure as the status handler of the calling thread until
/// the returned guard is dropped.
///
/// Status messages are free-form notes about the stage a computation is in
/// (as opposed to [progress](set_progress_handler) percentages). They are
/// emitted with [`status`]; returning [`ControlFlow::Break`] makes the
/// reporting call fail with
/// [`ErrorKind::Interrupted`](crate::ErrorKind::Interrupted).
///
/// Binds [`igraph_set_status_handler`](https://igraph.org/c/html/latest/igraph-Advanced.html#igraph_set_status_handler).
///
/// # Examples
///
/// ```
/// use igraph::misc;
/// use std::{cell::RefCell, ops::ControlFlow, rc::Rc};
///
/// let log = Rc::new(RefCell::new(String::new()));
/// let sink = Rc::clone(&log);
/// let _guard = misc::set_status_handler(move |msg| {
///     sink.borrow_mut().push_str(msg);
///     ControlFlow::Continue(())
/// });
/// misc::status("loading; ")?;
/// misc::status("done")?;
/// assert_eq!(*log.borrow(), "loading; done");
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn set_status_handler(
    handler: impl FnMut(&str) -> ControlFlow<()> + 'static,
) -> StatusHandlerGuard {
    install_status(Some(status_trampoline), Some(Box::new(handler)))
}

/// Installs igraph's predefined status handler, which writes the messages to
/// standard error, until the guard is dropped.
///
/// Binds [`igraph_status_handler_stderr`](https://igraph.org/c/html/latest/igraph-Advanced.html#igraph_status_handler_stderr).
pub fn set_status_handler_stderr() -> StatusHandlerGuard {
    install_status(Some(igraph_status_handler_stderr), None)
}

/// Runs `body` with `handler` installed as the status handler of the calling
/// thread, restoring the previous handler afterwards. See
/// [`set_status_handler`].
pub fn with_status_handler<R>(
    handler: impl FnMut(&str) -> ControlFlow<()> + 'static,
    body: impl FnOnce() -> R,
) -> R {
    let _guard = set_status_handler(handler);
    body()
}

/// Sends a status message to the installed status handler of the calling
/// thread (a no-op when none is installed).
///
/// Binds [`igraph_status`](https://igraph.org/c/html/latest/igraph-Advanced.html#igraph_status).
///
/// # Errors
/// [`ErrorKind::Interrupted`](crate::ErrorKind::Interrupted) if the handler
/// asked to stop; [`ErrorKind::InvalidValue`](crate::ErrorKind::InvalidValue)
/// if `message` contains a NUL byte.
pub fn status(message: &str) -> Result<()> {
    let message = to_cstring(message)?;
    igraph_call!(igraph_status(message.as_ptr(), std::ptr::null_mut()))
}

// ---------------------------------------------------------------------------
// Interruption
// ---------------------------------------------------------------------------

unsafe extern "C" fn interruption_trampoline() -> igraph_bool_t {
    in_finally_level(|| catch_panic_or(true, || call_with(&INTERRUPT, false, |f| f())))
}

fn set_interruption(c: igraph_interruption_handler_t) -> igraph_interruption_handler_t {
    unsafe { igraph_set_interruption_handler(c) }
}

/// Keeps an interruption handler installed; returned by
/// [`set_interruption_handler`].
///
/// Like [`ProgressHandlerGuard`], guards can be dropped in any order, and
/// each is bound to the creating thread.
#[must_use = "the handler is uninstalled as soon as the guard is dropped"]
pub struct InterruptionHandlerGuard {
    id: u64,
    _not_send: PhantomData<*const ()>,
}

impl std::fmt::Debug for InterruptionHandlerGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterruptionHandlerGuard")
            .finish_non_exhaustive()
    }
}

impl Drop for InterruptionHandlerGuard {
    fn drop(&mut self) {
        remove(&INTERRUPT, set_interruption, self.id);
    }
}

/// Installs a Rust closure as the interruption handler of the calling thread
/// until the returned guard is dropped.
///
/// igraph calls the interruption handler regularly during long computations
/// (shortest paths, SIR simulations, layouts, random graph generators, ...).
/// The closure returns `true` to request the interruption of the running
/// computation, which then fails with
/// [`ErrorKind::Interrupted`](crate::ErrorKind::Interrupted), and `false` to
/// let it go on. Typical uses are timeouts, cancellation flags set by
/// another thread (e.g. an `Arc<AtomicBool>`), or Ctrl-C handling. A panic
/// in the closure interrupts the computation and is resumed afterwards.
///
/// The closure may call igraph, but igraph polls the handler from inside
/// running computations, including the ARPACK eigensolvers (e.g.
/// [`linalg::arpack_rssolve`](crate::linalg::arpack_rssolve) or
/// [`Graph::eigenvector_centrality`](crate::Graph::eigenvector_centrality)).
/// igraph's ARPACK is not re-entrant (it keeps its iteration state in
/// thread-local statics), so a handler that may run during an ARPACK-based
/// computation must not start an ARPACK-based computation of its own: every
/// ARPACK-based function of the crate detects this and fails with an
/// [`ErrorKind::Failure`](crate::ErrorKind::Failure) error instead of
/// corrupting the running solver (see [`linalg`](crate::linalg)). The same
/// holds for progress handlers.
///
/// There is no default interruption handler. Binds
/// `igraph_set_interruption_handler` (see `igraph_interrupt.h`).
///
/// # Examples
///
/// ```
/// use igraph::prelude::*;
/// use igraph::misc;
/// use std::time::{Duration, Instant};
///
/// let deadline = Instant::now() + Duration::from_secs(3600);
/// let _guard = misc::set_interruption_handler(move || Instant::now() > deadline);
/// // Plenty of time: the simulation completes.
/// let g = Graph::from_edges(&[(0, 1), (1, 2)], 3, false)?;
/// assert_eq!(g.sir(1.0, 1.0, 3)?.len(), 3);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn set_interruption_handler(
    handler: impl FnMut() -> bool + 'static,
) -> InterruptionHandlerGuard {
    let handler: Box<InterruptDyn> = Box::new(handler);
    InterruptionHandlerGuard {
        id: push(
            &INTERRUPT,
            set_interruption,
            Some(interruption_trampoline),
            Some(handler),
        ),
        _not_send: PhantomData,
    }
}

/// Runs `body` with `handler` installed as the interruption handler of the
/// calling thread, restoring the previous handler afterwards. See
/// [`set_interruption_handler`].
///
/// # Examples
///
/// ```
/// use igraph::prelude::*;
/// use igraph::misc;
///
/// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], 3, false)?;
/// let err = misc::with_interruption_handler(|| true, || g.sir(1.0, 1.0, 10)).unwrap_err();
/// assert_eq!(err.kind(), ErrorKind::Interrupted);
/// # Ok::<(), igraph::Error>(())
/// ```
pub fn with_interruption_handler<R>(
    handler: impl FnMut() -> bool + 'static,
    body: impl FnOnce() -> R,
) -> R {
    let _guard = set_interruption_handler(handler);
    body()
}

/// Asks the installed interruption handler whether the current computation
/// should be interrupted; `false` when no handler is installed.
///
/// This is what igraph functions do internally; Rust code built on this
/// crate can call it to honor the same cancellation requests.
///
/// Binds `igraph_allow_interruption` (see `igraph_interrupt.h`).
pub fn allow_interruption() -> bool {
    ensure_init();
    let interrupt = unsafe { igraph_allow_interruption() };
    // A panicking handler was caught in the trampoline: resume it here.
    resume_panic();
    interrupt
}
