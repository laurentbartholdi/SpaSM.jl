#ifndef _FFLAS_FFPACK_CONFIG_H
#define _FFLAS_FFPACK_CONFIG_H 1
 
/* fflas-ffpack/config.h. Generated automatically at end of configure. */
/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef __FFLASFFPACK_AC_APPLE_UNIVERSAL_BUILD */

/* Define whether to compile in DEBUG mode */
/* #undef __FFLASFFPACK_DEBUG */

/* Define that architecture uses big endian storage */
/* #undef __FFLASFFPACK_HAVE_BIG_ENDIAN */

/* Define if BLAS is available */
#ifndef __FFLASFFPACK_HAVE_BLAS 
#define __FFLASFFPACK_HAVE_BLAS  1 
#endif

/* Define if BLAS is a CBLAS */
#ifndef __FFLASFFPACK_HAVE_CBLAS 
#define __FFLASFFPACK_HAVE_CBLAS  1 
#endif

/* Define if C interface to LAPACK is available */
/* #undef __FFLASFFPACK_HAVE_CLAPACK */

/* define if the compiler supports basic C++11 syntax */
#ifndef __FFLASFFPACK_HAVE_CXX11 
#define __FFLASFFPACK_HAVE_CXX11  1 
#endif

/* Define to 1 if you have the <dlfcn.h> header file. */
#ifndef __FFLASFFPACK_HAVE_DLFCN_H 
#define __FFLASFFPACK_HAVE_DLFCN_H  1 
#endif

/* Define to 1 if you have the <float.h> header file. */
#ifndef __FFLASFFPACK_HAVE_FLOAT_H 
#define __FFLASFFPACK_HAVE_FLOAT_H  1 
#endif

/* Define that compiler allows int128_t types */
#ifndef __FFLASFFPACK_HAVE_INT128 
#define __FFLASFFPACK_HAVE_INT128  1 
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#ifndef __FFLASFFPACK_HAVE_INTTYPES_H 
#define __FFLASFFPACK_HAVE_INTTYPES_H  1 
#endif

/* Define if LAPACK is installed */
/* #undef __FFLASFFPACK_HAVE_LAPACK */

/* Define to 1 if you have the <limits.h> header file. */
#ifndef __FFLASFFPACK_HAVE_LIMITS_H 
#define __FFLASFFPACK_HAVE_LIMITS_H  1 
#endif

/* Define that architecture uses little endian storage */
#ifndef __FFLASFFPACK_HAVE_LITTLE_ENDIAN 
#define __FFLASFFPACK_HAVE_LITTLE_ENDIAN  1 
#endif

/* Define to 1 if you have the <memory.h> header file. */
#ifndef __FFLASFFPACK_HAVE_MEMORY_H 
#define __FFLASFFPACK_HAVE_MEMORY_H  1 
#endif

/* Define if we use MKL for blas/lapack */
/* #undef __FFLASFFPACK_HAVE_MKL */

/* Define to 1 if you have the <pthread.h> header file. */
#ifndef __FFLASFFPACK_HAVE_PTHREAD_H 
#define __FFLASFFPACK_HAVE_PTHREAD_H  1 
#endif

/* Define to 1 if you have the <stddef.h> header file. */
#ifndef __FFLASFFPACK_HAVE_STDDEF_H 
#define __FFLASFFPACK_HAVE_STDDEF_H  1 
#endif

/* Define to 1 if you have the <stdint.h> header file. */
#ifndef __FFLASFFPACK_HAVE_STDINT_H 
#define __FFLASFFPACK_HAVE_STDINT_H  1 
#endif

/* Define to 1 if you have the <stdlib.h> header file. */
#ifndef __FFLASFFPACK_HAVE_STDLIB_H 
#define __FFLASFFPACK_HAVE_STDLIB_H  1 
#endif

/* Define to 1 if you have the <strings.h> header file. */
#ifndef __FFLASFFPACK_HAVE_STRINGS_H 
#define __FFLASFFPACK_HAVE_STRINGS_H  1 
#endif

/* Define to 1 if you have the <string.h> header file. */
#ifndef __FFLASFFPACK_HAVE_STRING_H 
#define __FFLASFFPACK_HAVE_STRING_H  1 
#endif

/* Define to 1 if you have the <sys/stat.h> header file. */
#ifndef __FFLASFFPACK_HAVE_SYS_STAT_H 
#define __FFLASFFPACK_HAVE_SYS_STAT_H  1 
#endif

/* Define to 1 if you have the <sys/time.h> header file. */
#ifndef __FFLASFFPACK_HAVE_SYS_TIME_H 
#define __FFLASFFPACK_HAVE_SYS_TIME_H  1 
#endif

/* Define to 1 if you have the <sys/types.h> header file. */
#ifndef __FFLASFFPACK_HAVE_SYS_TYPES_H 
#define __FFLASFFPACK_HAVE_SYS_TYPES_H  1 
#endif

/* Define to 1 if you have the <unistd.h> header file. */
#ifndef __FFLASFFPACK_HAVE_UNISTD_H 
#define __FFLASFFPACK_HAVE_UNISTD_H  1 
#endif

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#ifndef __FFLASFFPACK_LT_OBJDIR 
#define __FFLASFFPACK_LT_OBJDIR  ".libs/" 
#endif

/* Sets the number of threads given to OpenBLAS (default is 1) */
/* #undef __FFLASFFPACK_OPENBLAS_NUM_THREADS */

/* Name of package */
#ifndef __FFLASFFPACK_PACKAGE 
#define __FFLASFFPACK_PACKAGE  "fflas-ffpack" 
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef __FFLASFFPACK_PACKAGE_BUGREPORT 
#define __FFLASFFPACK_PACKAGE_BUGREPORT  "ffpack-devel@googlegroups.com" 
#endif

/* Define to the full name of this package. */
#ifndef __FFLASFFPACK_PACKAGE_NAME 
#define __FFLASFFPACK_PACKAGE_NAME  "FFLAS-FFPACK" 
#endif

/* Define to the full name and version of this package. */
#ifndef __FFLASFFPACK_PACKAGE_STRING 
#define __FFLASFFPACK_PACKAGE_STRING  "FFLAS-FFPACK 2.5.0" 
#endif

/* Define to the one symbol short name of this package. */
#ifndef __FFLASFFPACK_PACKAGE_TARNAME 
#define __FFLASFFPACK_PACKAGE_TARNAME  "fflas-ffpack" 
#endif

/* Define to the home page for this package. */
#ifndef __FFLASFFPACK_PACKAGE_URL 
#define __FFLASFFPACK_PACKAGE_URL  "https://github.com/linbox-team/fflas-ffpack" 
#endif

/* Define to the version of this package. */
#ifndef __FFLASFFPACK_PACKAGE_VERSION 
#define __FFLASFFPACK_PACKAGE_VERSION  "2.5.0" 
#endif

/* The size of `char', as computed by sizeof. */
#ifndef __FFLASFFPACK_SIZEOF_CHAR 
#define __FFLASFFPACK_SIZEOF_CHAR  1 
#endif

/* The size of `int', as computed by sizeof. */
#ifndef __FFLASFFPACK_SIZEOF_INT 
#define __FFLASFFPACK_SIZEOF_INT  4 
#endif

/* The size of `long', as computed by sizeof. */
#ifndef __FFLASFFPACK_SIZEOF_LONG 
#define __FFLASFFPACK_SIZEOF_LONG  8 
#endif

/* The size of `long long', as computed by sizeof. */
#ifndef __FFLASFFPACK_SIZEOF_LONG_LONG 
#define __FFLASFFPACK_SIZEOF_LONG_LONG  8 
#endif

/* The size of `short', as computed by sizeof. */
#ifndef __FFLASFFPACK_SIZEOF_SHORT 
#define __FFLASFFPACK_SIZEOF_SHORT  2 
#endif

/* The size of `__int64_t', as computed by sizeof. */
#ifndef __FFLASFFPACK_SIZEOF___INT64_T 
#define __FFLASFFPACK_SIZEOF___INT64_T  8 
#endif

/* Define to 1 if you have the ANSI C header files. */
#ifndef __FFLASFFPACK_STDC_HEADERS 
#define __FFLASFFPACK_STDC_HEADERS  1 
#endif

/* Define if OMP is available */
/* #undef __FFLASFFPACK_USE_OPENMP */

/* Version number of package */
#ifndef __FFLASFFPACK_VERSION 
#define __FFLASFFPACK_VERSION  "2.5.0" 
#endif

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif
 
/* once: _FFLAS_FFPACK_CONFIG_H */
#endif
