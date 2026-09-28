//! Error handling: turning igraph error codes into Rust [`Result`]s.
//!
//! The C library reports failures by returning an [`igraph_error_t`] code and
//! by invoking a (thread-local) *error handler*. The default handler aborts
//! the whole process, which is hardly what a Rust program wants. The first time
//! a thread calls into this crate, [`ensure_init`] installs a handler that
//! records the error message (so that it can be reported in an [`Error`]) and
//! then frees the current level of igraph's internal cleanup ("finally")
//! stack, exactly like `igraph_error_handler_ignore` does; the failing
//! function then returns its error code, which [`check`] converts into an
//! [`Error`].
//!
//! Warnings emitted by igraph are collected in a thread-local buffer as well;
//! retrieve them with [`take_warnings`].
//!
//! Rust callbacks invoked by igraph must never unwind into C: trampolines run
//! them inside [`catch_panic`] (or [`catch_panic_or`]), which stores the
//! panic and returns an error code to igraph; the wrapper's [`check`] then
//! resumes the panic in Rust, after igraph has cleaned up.
//!
//! # Threads
//!
//! The crate requires igraph to be built in thread-safe mode (the
//! `IGRAPH_THREAD_SAFE` macro of `igraph_threading.h` must be `1`, which is
//! checked at compile time): every thread then has its own igraph state,
//! namely error and warning handlers, "finally" stack (the list of
//! temporary objects to free if the running function fails) and default
//! random number generator. [`ensure_init`] sets them up the first time a
//! thread calls into the crate, so igraph can be used from many threads at
//! once. Nothing is shared between threads except what the user shares:
//! a [`Graph`](crate::Graph) is [`Send`] but not [`Sync`], and seeding the
//! random number generator (see [`rng`](crate::rng)) affects the calling
//! thread only.
//!
//! # Callbacks that call igraph
//!
//! A Rust closure invoked by a running igraph function (a visitor, a clique
//! or motif handler, ...) may itself call igraph functions, for instance to
//! run a nested search. Two consequences of igraph's per-thread "finally"
//! stack are handled here:
//!
//! - [`catch_panic`] and [`catch_panic_or`] run the closure in a fresh level
//!   of the finally stack (`IGRAPH_FINALLY_ENTER` / `IGRAPH_FINALLY_EXIT`),
//!   so an igraph call that fails inside the closure frees only its own
//!   temporaries, never those of the outer function that is still running;
//!   the closure receives an ordinary [`Err`] and may ignore it.
//! - The finally stack holds at most 100 entries per thread, and igraph
//!   aborts the process when it overflows. Each nested call adds its own
//!   entries on top of the outer ones, so [`igraph_call!`](crate::igraph_call)
//!   (and [`Graph::init_with`](crate::Graph::init_with)) refuse to start a C
//!   call when the stack already holds more than
//!   [`MAX_NESTED_FINALLY_ENTRIES`] entries, returning an
//!   [`ErrorKind::Failure`] error ("igraph calls nested too deeply") instead;
//!   in practice this allows a dozen or more levels of nested searches.
//!
//! | C (`igraph_error.h`) | Rust |
//! |-------------------|------|
//! | `igraph_error_t` return codes | [`Result`], [`Error`], [`ErrorKind`] ([`ErrorKind::from_raw`], [`ErrorKind::to_raw`]) |
//! | `igraph_strerror` | [`strerror`], [`ErrorKind::description`], [`Display`](std::fmt::Display) |
//! | `igraph_set_error_handler`, `igraph_set_warning_handler`, `igraph_setup` | [`ensure_init`] (automatic), [`is_initialized`] |
//! | `IGRAPH_CHECK` | [`igraph_call!`](crate::igraph_call), [`check`] |
//! | warnings (`IGRAPH_WARNING`) | [`take_warnings`] |
//! | callbacks returning `IGRAPH_INTERRUPTED` | [`catch_panic`], [`catch_panic_or`], [`resume_panic`], [`has_pending_panic`] |
//! | `IGRAPH_FINALLY_ENTER`, `IGRAPH_FINALLY_EXIT`, `IGRAPH_FINALLY_STACK_SIZE` | used by [`catch_panic`], [`catch_panic_or`] and [`igraph_call!`](crate::igraph_call); [`finally_stack_size`] |
//!
//! Not wrapped: the error and warning *reporting* functions (`igraph_error`,
//! `igraph_errorf`, `igraph_errorvf`, `igraph_warning`, `igraph_warningf`,
//! `igraph_fatal`, `igraph_fatalf`) are meant for C code that raises igraph
//! errors, while Rust code returns an [`Error`] (see [`Error::new`]).
//! `igraph_set_fatal_handler` is deliberately left alone: a fatal handler
//! must not return, and a Rust function cannot unwind through the C frames
//! above it, so igraph's default handler, which aborts the process, stays in
//! place. Fatal errors signal broken internal invariants (e.g. a corrupt
//! finally stack), not ordinary failures.
//!
//! See also [`misc::set_progress_handler`](crate::misc::set_progress_handler),
//! [`misc::set_status_handler`](crate::misc::set_status_handler) and
//! [`misc::set_interruption_handler`](crate::misc::set_interruption_handler)
//! for igraph's other (progress, status and interruption) handlers.
//!
//! # Example
//!
//! ```
//! use igraph::prelude::*;
//!
//! let mut g = Graph::new(3, false);
//! // Vertex 7 does not exist: igraph reports an "invalid vertex id" error.
//! let err = g.add_edge(0, 7).unwrap_err();
//! assert_eq!(err.kind(), ErrorKind::InvalidVertexId);
//! assert!(!err.message().is_empty());
//! // The error names the C source location that raised it.
//! assert!(err.file().ends_with(".c") && err.line() > 0);
//! // Errors compose with `?` and `std::error::Error`.
//! fn degree_of_7(g: &Graph) -> Result<i64> {
//!     g.degree_of(7, NeighborMode::All, Loops::Twice)
//! }
//! let boxed: Box<dyn std::error::Error> = degree_of_7(&g).unwrap_err().into();
//! assert!(boxed.to_string().starts_with(&ErrorKind::InvalidVertexId.description()));
//! ```

use crate::ffi::*;
use std::{
    any::Any,
    cell::{Cell, RefCell},
    ffi::{CStr, c_char, c_int},
    fmt,
};

/// The kind of an igraph error, mirroring the C enumeration
/// [`igraph_error_type_t`](https://igraph.org/c/html/latest/igraph-Error.html#igraph_error_type_t).
///
/// The error codes removed in igraph 1.0 (`IGRAPH_EINVEVECTOR`,
/// `IGRAPH_NONSQUARE`, `IGRAPH_EDIVZERO`, `IGRAPH_EATTRIBUTES`,
/// `IGRAPH_ELAPACK`, `IGRAPH_EDRL`, `IGRAPH_EGLP` and the `IGRAPH_GLP_*`
/// codes, `IGRAPH_CPUTIME`) have no variant, nor do the ARPACK-specific codes
/// that 1.0 moved to `igraph_arpack_error_t`; an unknown code maps to
/// [`ErrorKind::Other`]. Code 8, `IGRAPH_NONSQUARE` before 1.0, is now
/// `IGRAPH_EINVEID` ([`ErrorKind::InvalidEdgeId`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Generic failure (`IGRAPH_FAILURE`).
    Failure,
    /// Out of memory (`IGRAPH_ENOMEM`).
    OutOfMemory,
    /// Parse error while reading a file (`IGRAPH_PARSEERROR`).
    Parse,
    /// Invalid value, argument or combination of arguments (`IGRAPH_EINVAL`).
    InvalidValue,
    /// Object already exists (`IGRAPH_EXISTS`).
    Exists,
    /// Invalid vertex id (`IGRAPH_EINVVID`).
    InvalidVertexId,
    /// Invalid edge id (`IGRAPH_EINVEID`).
    InvalidEdgeId,
    /// Invalid neighbor mode (`IGRAPH_EINVMODE`).
    InvalidMode,
    /// File operation failed (`IGRAPH_EFILE`).
    File,
    /// Functionality not implemented (`IGRAPH_UNIMPLEMENTED`).
    Unimplemented,
    /// Computation interrupted, e.g. by a callback (`IGRAPH_INTERRUPTED`).
    Interrupted,
    /// Numeric procedure did not converge (`IGRAPH_DIVERGED`).
    Diverged,
    /// ARPACK error (`IGRAPH_EARPACK`).
    Arpack,
    /// Negative cycle detected in shortest path computation (`IGRAPH_ENEGCYCLE`).
    NegativeCycle,
    /// Internal igraph error, likely a bug (`IGRAPH_EINTERNAL`).
    Internal,
    /// Attribute combination error (`IGRAPH_EATTRCOMBINE`).
    AttributeCombination,
    /// Integer or double overflow (`IGRAPH_EOVERFLOW`).
    Overflow,
    /// Integer or double underflow (`IGRAPH_EUNDERFLOW`).
    Underflow,
    /// A random walk got stuck (`IGRAPH_ERWSTUCK`).
    RandomWalkStuck,
    /// Stop requested, e.g. by a callback (`IGRAPH_STOP`).
    Stop,
    /// Maximum vertex or edge count exceeded (`IGRAPH_ERANGE`).
    Range,
    /// The input problem has no solution (`IGRAPH_ENOSOL`).
    NoSolution,
    /// An error code not known to this version of the bindings.
    Other(u32),
    /// A Rust callback panicked while igraph was running; the panic is
    /// resumed by the wrapper, so this kind is only observable in `catch_unwind`.
    CallbackPanic,
}

impl ErrorKind {
    /// Maps a raw igraph error code to an [`ErrorKind`].
    pub fn from_raw(code: igraph_error_t) -> Self {
        match code {
            igraph_error_type_t_IGRAPH_FAILURE => Self::Failure,
            igraph_error_type_t_IGRAPH_ENOMEM => Self::OutOfMemory,
            igraph_error_type_t_IGRAPH_PARSEERROR => Self::Parse,
            igraph_error_type_t_IGRAPH_EINVAL => Self::InvalidValue,
            igraph_error_type_t_IGRAPH_EXISTS => Self::Exists,
            igraph_error_type_t_IGRAPH_EINVVID => Self::InvalidVertexId,
            igraph_error_type_t_IGRAPH_EINVEID => Self::InvalidEdgeId,
            igraph_error_type_t_IGRAPH_EINVMODE => Self::InvalidMode,
            igraph_error_type_t_IGRAPH_EFILE => Self::File,
            igraph_error_type_t_IGRAPH_UNIMPLEMENTED => Self::Unimplemented,
            igraph_error_type_t_IGRAPH_INTERRUPTED => Self::Interrupted,
            igraph_error_type_t_IGRAPH_DIVERGED => Self::Diverged,
            igraph_error_type_t_IGRAPH_EARPACK => Self::Arpack,
            igraph_error_type_t_IGRAPH_ENEGCYCLE => Self::NegativeCycle,
            igraph_error_type_t_IGRAPH_EINTERNAL => Self::Internal,
            igraph_error_type_t_IGRAPH_EATTRCOMBINE => Self::AttributeCombination,
            igraph_error_type_t_IGRAPH_EOVERFLOW => Self::Overflow,
            igraph_error_type_t_IGRAPH_EUNDERFLOW => Self::Underflow,
            igraph_error_type_t_IGRAPH_ERWSTUCK => Self::RandomWalkStuck,
            igraph_error_type_t_IGRAPH_STOP => Self::Stop,
            igraph_error_type_t_IGRAPH_ERANGE => Self::Range,
            igraph_error_type_t_IGRAPH_ENOSOL => Self::NoSolution,
            other => Self::Other(other),
        }
    }

    /// The raw igraph error code corresponding to this kind (the inverse of
    /// [`from_raw`](Self::from_raw)).
    ///
    /// [`ErrorKind::CallbackPanic`] maps to `IGRAPH_INTERRUPTED`, the code
    /// returned to igraph when a Rust callback panics.
    ///
    /// ```
    /// use igraph::prelude::*;
    /// for kind in [ErrorKind::InvalidValue, ErrorKind::Parse, ErrorKind::Stop] {
    ///     assert_eq!(ErrorKind::from_raw(kind.to_raw()), kind);
    /// }
    /// ```
    pub fn to_raw(self) -> igraph_error_t {
        match self {
            Self::Failure => igraph_error_type_t_IGRAPH_FAILURE,
            Self::OutOfMemory => igraph_error_type_t_IGRAPH_ENOMEM,
            Self::Parse => igraph_error_type_t_IGRAPH_PARSEERROR,
            Self::InvalidValue => igraph_error_type_t_IGRAPH_EINVAL,
            Self::Exists => igraph_error_type_t_IGRAPH_EXISTS,
            Self::InvalidVertexId => igraph_error_type_t_IGRAPH_EINVVID,
            Self::InvalidEdgeId => igraph_error_type_t_IGRAPH_EINVEID,
            Self::InvalidMode => igraph_error_type_t_IGRAPH_EINVMODE,
            Self::File => igraph_error_type_t_IGRAPH_EFILE,
            Self::Unimplemented => igraph_error_type_t_IGRAPH_UNIMPLEMENTED,
            Self::Interrupted | Self::CallbackPanic => igraph_error_type_t_IGRAPH_INTERRUPTED,
            Self::Diverged => igraph_error_type_t_IGRAPH_DIVERGED,
            Self::Arpack => igraph_error_type_t_IGRAPH_EARPACK,
            Self::NegativeCycle => igraph_error_type_t_IGRAPH_ENEGCYCLE,
            Self::Internal => igraph_error_type_t_IGRAPH_EINTERNAL,
            Self::AttributeCombination => igraph_error_type_t_IGRAPH_EATTRCOMBINE,
            Self::Overflow => igraph_error_type_t_IGRAPH_EOVERFLOW,
            Self::Underflow => igraph_error_type_t_IGRAPH_EUNDERFLOW,
            Self::RandomWalkStuck => igraph_error_type_t_IGRAPH_ERWSTUCK,
            Self::Stop => igraph_error_type_t_IGRAPH_STOP,
            Self::Range => igraph_error_type_t_IGRAPH_ERANGE,
            Self::NoSolution => igraph_error_type_t_IGRAPH_ENOSOL,
            Self::Other(code) => code,
        }
    }

    /// igraph's textual description of this kind of error
    /// ([`igraph_strerror`](https://igraph.org/c/html/latest/igraph-Error.html#igraph_strerror)),
    /// e.g. `"Invalid value"` for [`ErrorKind::InvalidValue`].
    pub fn description(self) -> String {
        strerror(self.to_raw())
    }
}

impl fmt::Display for ErrorKind {
    /// Writes igraph's description of the error kind, see [`ErrorKind::description`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.description())
    }
}

/// igraph's textual description of a raw error code
/// ([`igraph_strerror`](https://igraph.org/c/html/latest/igraph-Error.html#igraph_strerror)).
///
/// ```
/// use igraph::{error::strerror, ffi};
/// assert_eq!(strerror(ffi::igraph_error_type_t_IGRAPH_SUCCESS), "No error");
/// assert!(!strerror(ffi::igraph_error_type_t_IGRAPH_EINVVID).is_empty());
/// ```
pub fn strerror(code: igraph_error_t) -> String {
    let ptr = unsafe { igraph_strerror(code) };
    lossy(ptr)
}

/// An error reported by the igraph C library, or raised by a wrapper that
/// validated its arguments on the Rust side.
///
/// It carries the [`ErrorKind`], the raw code, and the message, source file
/// and line recorded by igraph's error handler (for errors raised in Rust,
/// the file is empty and the line is `0`). [`Display`](fmt::Display) prints
/// `<description>: <message> (<file>:<line>)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    code: igraph_error_t,
    message: String,
    file: String,
    line: i32,
}

impl Error {
    /// Builds an error from its parts; mainly useful to wrappers that validate
    /// arguments on the Rust side before calling into igraph.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        // Every kind maps to its own igraph code, so that `code()` and the
        // `Display` description are consistent with `kind()`.
        let code = kind.to_raw();
        Self {
            kind,
            code,
            message: message.into(),
            file: String::new(),
            line: 0,
        }
    }

    /// Shorthand for an [`ErrorKind::InvalidValue`] error raised on the Rust side.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidValue, message)
    }

    /// The kind of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The raw igraph error code.
    pub fn code(&self) -> igraph_error_t {
        self.code
    }

    /// The human readable message reported by igraph.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The C source file where the error was raised (empty if unknown).
    pub fn file(&self) -> &str {
        &self.file
    }

    /// The C source line where the error was raised (`0` if unknown).
    pub fn line(&self) -> i32 {
        self.line
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", strerror(self.code))?;
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        if !self.file.is_empty() {
            write!(f, " ({}:{})", self.file, self.line)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

/// Specialized [`Result`](std::result::Result) type used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

struct Recorded {
    message: String,
    file: String,
    line: i32,
}

thread_local! {
    static INITIALIZED: Cell<bool> = const { Cell::new(false) };
    static LAST_ERROR: RefCell<Option<Recorded>> = const { RefCell::new(None) };
    static WARNINGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static PANIC: RefCell<Option<Box<dyn Any + Send + 'static>>> = const { RefCell::new(None) };
}

/// Maximum number of warnings kept in the thread-local buffer.
const MAX_WARNINGS: usize = 1024;

// The per-thread model described in the module docs needs a thread-safe
// igraph build (thread-local handlers, finally stack and RNG pointer).
const _: () = assert!(
    IGRAPH_THREAD_SAFE == 1,
    "igraph must be built in thread-safe mode (IGRAPH_THREAD_SAFE=1)"
);

/// Size of igraph's per-thread finally stack (fixed in `error.c`); igraph
/// aborts the process when a function needs more entries.
const FINALLY_STACK_CAPACITY: c_int = 100;

/// The number of entries of igraph's (per-thread) finally stack above which
/// [`igraph_call!`](crate::igraph_call) and
/// [`Graph::init_with`](crate::Graph::init_with) refuse to start a C call,
/// see the [module docs](self#callbacks-that-call-igraph).
///
/// The stack holds 100 entries; keeping half of them free leaves room for
/// the temporaries of any single igraph function (and its callbacks).
pub const MAX_NESTED_FINALLY_ENTRIES: usize = 50;

/// The current number of entries of igraph's finally stack on the calling
/// thread (`IGRAPH_FINALLY_STACK_SIZE`, undocumented in `igraph_error.h`).
///
/// It is zero outside of igraph calls; inside a callback it counts the
/// temporaries of the igraph functions that are running.
///
/// ```
/// assert_eq!(igraph::error::finally_stack_size(), 0);
/// ```
pub fn finally_stack_size() -> usize {
    // SAFETY: reads a thread-local counter.
    let size = unsafe { IGRAPH_FINALLY_STACK_SIZE() };
    usize::try_from(size).unwrap_or(0)
}

/// Converts a Rust size (length, capacity, count) into an `igraph_int_t`.
///
/// Sizes above `igraph_int_t::MAX` would turn negative, which igraph either
/// rejects with an assertion (aborting the process) or, for some containers
/// such as bitsets, silently accepts and turns into out of bounds accesses.
///
/// # Panics
/// Like [`Vec::with_capacity`], with "capacity overflow" for such sizes.
#[track_caller]
pub(crate) fn int_size(n: usize) -> igraph_int_t {
    match igraph_int_t::try_from(n) {
        Ok(n) => n,
        Err(_) => panic!("capacity overflow: {n} does not fit in an igraph_int_t"),
    }
}

fn lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

unsafe extern "C" fn record_error(
    reason: *const c_char,
    file: *const c_char,
    line: c_int,
    errno: igraph_error_t,
) {
    let recorded = Recorded {
        message: lossy(reason),
        file: lossy(file),
        line,
    };
    // As an error propagates, igraph re-raises it with an empty message:
    // keep the first, most informative, record (the root cause).
    // `try_with`: this runs inside C code, where a panic would abort; during
    // thread teardown the thread-local may be gone, then we record nothing.
    let _ = LAST_ERROR.try_with(|e| {
        let Ok(mut e) = e.try_borrow_mut() else {
            return;
        };
        if e.as_ref()
            .is_none_or(|old| old.message.is_empty() && !recorded.message.is_empty())
        {
            *e = Some(recorded);
        }
    });
    // Frees the objects on igraph's "finally" stack, as the stock handler does.
    unsafe { igraph_error_handler_ignore(reason, file, line, errno) };
}

unsafe extern "C" fn record_warning(reason: *const c_char, file: *const c_char, line: c_int) {
    let warning = format!("{} ({}:{})", lossy(reason), lossy(file), line);
    let _ = WARNINGS.try_with(|w| {
        if let Ok(mut w) = w.try_borrow_mut()
            && w.len() < MAX_WARNINGS
        {
            w.push(warning);
        }
    });
}

/// Initializes igraph for the calling thread, once.
///
/// It installs the error and warning handlers of this crate
/// ([`igraph_set_error_handler`](https://igraph.org/c/html/latest/igraph-Error.html#igraph_set_error_handler),
/// [`igraph_set_warning_handler`](https://igraph.org/c/html/latest/igraph-Error.html#igraph_set_warning_handler)), gives the
/// thread its own randomly seeded default random number generator (in
/// igraph 1.0.0 and 1.0.1 the default generator *pointer* is thread-local,
/// but it initially points to a single global generator shared by all
/// threads, which would make concurrent random functions race; see
/// [`rng`](crate::rng)), and calls `igraph_setup()`.
/// Every safe wrapper calls it (it's cheap after the first time), so users
/// rarely need to call it explicitly. It is idempotent.
pub fn ensure_init() {
    // `try_with`: during thread teardown (e.g. when a value owning igraph
    // memory is dropped by another thread-local destructor) the flag may be
    // gone; the handlers were installed long before, so there is nothing to do.
    let _ = INITIALIZED.try_with(|init| {
        if !init.get() {
            init.set(true);
            unsafe {
                igraph_set_error_handler(Some(record_error));
                igraph_set_warning_handler(Some(record_warning));
            }
            // igraph's default RNG *pointer* is thread-local, but it points to
            // one global generator shared by all threads: using it from two
            // threads at once would be a data race. Give each thread its own.
            crate::rng::install_thread_default_rng();
            // Seeds the default generator only if not seeded yet: the
            // per-thread one already is, so the shared global is not touched.
            unsafe { igraph_setup() };
        }
    });
}

/// Whether [`ensure_init`] already ran on the calling thread.
pub fn is_initialized() -> bool {
    INITIALIZED.try_with(Cell::get).unwrap_or(false)
}

/// Forgets any error recorded by the handler and not yet consumed by
/// [`check`]. Called by [`igraph_call!`] before every C call, so that a stale
/// record left by an unchecked call can never be attached to a later error.
#[doc(hidden)]
pub fn reset_last_error() {
    let _ = LAST_ERROR.try_with(|e| {
        if let Ok(mut e) = e.try_borrow_mut() {
            *e = None;
        }
    });
}

/// Converts a raw igraph return code into a [`Result`], attaching the message
/// recorded by the error handler: `IGRAPH_SUCCESS` gives `Ok(())`, any
/// other code an [`Error`] of the corresponding [`ErrorKind`].
///
/// If a Rust callback panicked during the call (see [`catch_panic`]), the
/// panic is resumed here.
pub fn check(code: igraph_error_t) -> Result<()> {
    // Consume the record first: resuming a panic unwinds out of this
    // function, and a stale record must not leak into the next error.
    let recorded = LAST_ERROR
        .try_with(|e| e.borrow_mut().take())
        .ok()
        .flatten();
    resume_panic();
    if code == igraph_error_type_t_IGRAPH_SUCCESS {
        return Ok(());
    }
    let (message, file, line) = match recorded {
        Some(Recorded {
            message,
            file,
            line,
        }) => (message, file, line),
        None => (String::new(), String::new(), 0),
    };
    Err(Error {
        kind: ErrorKind::from_raw(code),
        code,
        message,
        file,
        line,
    })
}

/// Calls an igraph function returning an [`igraph_error_t`] and converts the
/// outcome into a [`Result`]`<()>`, making sure the calling thread is
/// initialized first (see [`ensure_init`]).
///
/// ```
/// use igraph::{ffi, igraph_call};
/// let mut v = std::mem::MaybeUninit::<ffi::igraph_vector_int_t>::uninit();
/// igraph_call!(ffi::igraph_vector_int_init(v.as_mut_ptr(), 3)).unwrap();
/// let v = unsafe { v.assume_init() }; // dropped (and destroyed) at scope exit
/// assert_eq!(v.len(), 3);
/// ```
#[macro_export]
macro_rules! igraph_call {
    ($call:expr) => {{
        match $crate::error::prepare_call() {
            ::std::result::Result::Ok(()) => {
                // Running the given raw FFI call inside `unsafe` is the
                // documented purpose of this macro.
                #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
                let code = unsafe { $call };
                $crate::error::check(code)
            }
            ::std::result::Result::Err(e) => ::std::result::Result::Err(e),
        }
    }};
}

/// Prepares the calling thread for a C call: [`ensure_init`], then
/// [`reset_last_error`], then refuses (with [`ErrorKind::Failure`]) when the
/// finally stack holds more than [`MAX_NESTED_FINALLY_ENTRIES`] entries, as
/// the call could overflow it and make igraph abort the process. Called by
/// [`igraph_call!`](crate::igraph_call) before every call.
#[doc(hidden)]
pub fn prepare_call() -> Result<()> {
    ensure_init();
    reset_last_error();
    let size = finally_stack_size();
    if size > MAX_NESTED_FINALLY_ENTRIES {
        return Err(Error::new(
            ErrorKind::Failure,
            format!(
                "igraph calls nested too deeply inside callbacks: igraph's finally stack \
                 already holds {size} of its {FINALLY_STACK_CAPACITY} entries"
            ),
        ));
    }
    Ok(())
}

/// Drains and returns the warnings that igraph emitted on the calling thread
/// (as `"<message> (<file>:<line>)"`), oldest first.
///
/// igraph reports non-fatal problems (e.g. a numerical method that did not
/// fully converge) through its warning handler; this crate buffers them per
/// thread (at most 1024, the later ones are dropped) instead of printing
/// them to standard error like igraph's default handler does.
pub fn take_warnings() -> Vec<String> {
    WARNINGS.with(|w| std::mem::take(&mut *w.borrow_mut()))
}

/// Runs a Rust closure invoked *from C* (a callback), catching panics so that
/// they never unwind across the FFI boundary.
///
/// If `f` panics, the payload is stored and `IGRAPH_INTERRUPTED` is returned
/// to igraph, which stops the computation; the next [`check`] on the same
/// thread then resumes the panic in Rust code. Use it in every
/// `extern "C" fn` trampoline that calls user code:
///
/// ```ignore
/// unsafe extern "C" fn trampoline(/* ... */, extra: *mut c_void) -> igraph_error_t {
///     catch_panic(|| { /* call the user closure */ igraph_error_type_t_IGRAPH_SUCCESS })
/// }
/// ```
///
/// `f` runs in a new level of igraph's finally stack
/// (`IGRAPH_FINALLY_ENTER` / `IGRAPH_FINALLY_EXIT`): when an igraph call
/// made by `f` fails, the error handler frees the temporaries of that call
/// only, not those of the igraph function that invoked the callback and is
/// still running (see the [module docs](self#callbacks-that-call-igraph)).
///
/// ```
/// use igraph::{error::{catch_panic, check}, ffi};
/// // Outside of a callback the finally stack is empty, and stays so.
/// let code = catch_panic(|| {
///     // A failing igraph call inside the "callback" is an ordinary `Err`.
///     assert!(igraph::prelude::Graph::new(2, false).add_edge(0, 5).is_err());
///     ffi::igraph_error_type_t_IGRAPH_SUCCESS
/// });
/// assert!(check(code).is_ok());
/// assert_eq!(igraph::error::finally_stack_size(), 0);
/// ```
pub fn catch_panic(f: impl FnOnce() -> igraph_error_t) -> igraph_error_t {
    catch_panic_or(igraph_error_type_t_IGRAPH_INTERRUPTED, f)
}

/// Like [`catch_panic`], for callbacks whose C signature returns a value
/// other than an error code (e.g. a boolean): `on_panic` is returned to C.
///
/// Like [`catch_panic`], it runs `f` in a new level of igraph's finally
/// stack.
pub fn catch_panic_or<T>(on_panic: T, f: impl FnOnce() -> T) -> T {
    // SAFETY (for the finally-stack calls below): bookkeeping on igraph's
    // thread-local finally stack. The entries below `entry_size` belong to
    // the running igraph function(s) and are at the current level or lower,
    // so a new level can be opened; the matching EXIT always runs, since
    // `catch_unwind` returns on panic too.
    let entry_size = unsafe { IGRAPH_FINALLY_STACK_SIZE() };
    unsafe { IGRAPH_FINALLY_ENTER() };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    // Every igraph call made by `f` has returned by now, and each of them
    // either popped its entries or (on failure) had the error handler free
    // them. Entries left over would come from a C function that returned an
    // error without reporting it: they point into stack frames that no
    // longer exist, so drop them *without* running their destructors (a
    // leak at worst) rather than leave them for a later error to "free".
    let left_over = unsafe { IGRAPH_FINALLY_STACK_SIZE() } - entry_size;
    if left_over > 0 {
        unsafe { IGRAPH_FINALLY_CLEAN(left_over) };
    }
    unsafe { IGRAPH_FINALLY_EXIT() };
    match outcome {
        Ok(value) => value,
        Err(payload) => {
            store_panic(payload);
            on_panic
        }
    }
}

/// Stores the payload of a caught panic; the *first* panic wins, later ones
/// (e.g. from callbacks igraph keeps calling) are dropped.
fn store_panic(payload: Box<dyn Any + Send + 'static>) {
    let _ = PANIC.try_with(|p| {
        if let Ok(mut p) = p.try_borrow_mut()
            && p.is_none()
        {
            *p = Some(payload);
        }
    });
}

/// Resumes a panic caught by [`catch_panic`] on this thread, if any.
pub fn resume_panic() {
    if let Some(payload) = PANIC.try_with(|p| p.borrow_mut().take()).ok().flatten() {
        std::panic::resume_unwind(payload);
    }
}

/// Whether a panic caught by [`catch_panic`] is waiting to be resumed on the
/// calling thread (by the next [`check`] or [`resume_panic`]).
///
/// Trampolines of callbacks that igraph keeps calling after an error can use
/// it to skip the user code once it has panicked.
pub fn has_pending_panic() -> bool {
    PANIC.try_with(|p| p.borrow().is_some()).unwrap_or(false)
}
