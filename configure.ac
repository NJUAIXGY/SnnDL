AC_INIT([SnnDL], [1.0.0], [your-email@domain.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

# Check for C++ compiler with C++17 support
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX([17], [noext], [mandatory])

# Check for libtool
LT_INIT

# Check for SST Core
PKG_CHECK_MODULES([SST_CORE], [SSTCore >= 15.0.0])

# Add debug option
AC_ARG_ENABLE([debug],
    AS_HELP_STRING([--enable-debug], [Enable debug build]),
    [debug="$enableval"],
    [debug=no])

if test "x$debug" = "xyes"; then
    CXXFLAGS="$CXXFLAGS -g -O0 -DDEBUG"
else
    CXXFLAGS="$CXXFLAGS -O3 -DNDEBUG"
fi

# Add verbose option
AC_ARG_ENABLE([verbose],
    AS_HELP_STRING([--enable-verbose], [Enable verbose logging]),
    [verbose="$enableval"],
    [verbose=no])

if test "x$verbose" = "xyes"; then
    CXXFLAGS="$CXXFLAGS -DVERBOSE"
fi

# Output files
AC_CONFIG_FILES([
    Makefile
])

AC_OUTPUT

# Summary
echo "
Configuration Summary:
  Package: $PACKAGE_STRING
  C++ Compiler: $CXX
  C++ Flags: $CXXFLAGS
  SST Core: $SST_CORE_VERSION
  Debug: $debug
  Verbose: $verbose
  Install prefix: $prefix
"
