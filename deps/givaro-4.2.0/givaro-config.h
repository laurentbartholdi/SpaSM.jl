#ifndef _GIVARO_CONFIG_H
#define _GIVARO_CONFIG_H 1
 
/* givaro-config.h. Generated automatically at end of configure. */
/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef __GIVARO_AC_APPLE_UNIVERSAL_BUILD */

/* Define whether to compile in DEBUG mode */
/* #undef __GIVARO_DEBUG */

/* Define that architecture uses big endian storage */
/* #undef __GIVARO_HAVE_BIG_ENDIAN */

/* define if the compiler supports basic C++11 syntax */
#ifndef __GIVARO_HAVE_CXX11 
#define __GIVARO_HAVE_CXX11  1 
#endif

/* Define to 1 if you have the <dlfcn.h> header file. */
#ifndef __GIVARO_HAVE_DLFCN_H 
#define __GIVARO_HAVE_DLFCN_H  1 
#endif

/* Define to 1 if you have the <float.h> header file. */
#ifndef __GIVARO_HAVE_FLOAT_H 
#define __GIVARO_HAVE_FLOAT_H  1 
#endif

/* Define if GMP is installed and OK */
#ifndef __GIVARO_HAVE_GMP 
#define __GIVARO_HAVE_GMP  1  
#endif

/* Define that compiler allows int128_t types */
#ifndef __GIVARO_HAVE_INT128 
#define __GIVARO_HAVE_INT128  1 
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#ifndef __GIVARO_HAVE_INTTYPES_H 
#define __GIVARO_HAVE_INTTYPES_H  1 
#endif

/* Define to 1 if you have the <limits.h> header file. */
#ifndef __GIVARO_HAVE_LIMITS_H 
#define __GIVARO_HAVE_LIMITS_H  1 
#endif

/* Define that architecture uses little endian storage */
#ifndef __GIVARO_HAVE_LITTLE_ENDIAN 
#define __GIVARO_HAVE_LITTLE_ENDIAN  1 
#endif

/* Define to 1 if you have the <memory.h> header file. */
#ifndef __GIVARO_HAVE_MEMORY_H 
#define __GIVARO_HAVE_MEMORY_H  1 
#endif

/* Define to 1 if you have the <pthread.h> header file. */
#ifndef __GIVARO_HAVE_PTHREAD_H 
#define __GIVARO_HAVE_PTHREAD_H  1 
#endif

/* Define to 1 if you have the <stddef.h> header file. */
#ifndef __GIVARO_HAVE_STDDEF_H 
#define __GIVARO_HAVE_STDDEF_H  1 
#endif

/* Define to 1 if you have the <stdint.h> header file. */
#ifndef __GIVARO_HAVE_STDINT_H 
#define __GIVARO_HAVE_STDINT_H  1 
#endif

/* Define to 1 if you have the <stdlib.h> header file. */
#ifndef __GIVARO_HAVE_STDLIB_H 
#define __GIVARO_HAVE_STDLIB_H  1 
#endif

/* Define to 1 if you have the <strings.h> header file. */
#ifndef __GIVARO_HAVE_STRINGS_H 
#define __GIVARO_HAVE_STRINGS_H  1 
#endif

/* Define to 1 if you have the <string.h> header file. */
#ifndef __GIVARO_HAVE_STRING_H 
#define __GIVARO_HAVE_STRING_H  1 
#endif

/* Define to 1 if you have the <sys/stat.h> header file. */
#ifndef __GIVARO_HAVE_SYS_STAT_H 
#define __GIVARO_HAVE_SYS_STAT_H  1 
#endif

/* Define to 1 if you have the <sys/time.h> header file. */
#ifndef __GIVARO_HAVE_SYS_TIME_H 
#define __GIVARO_HAVE_SYS_TIME_H  1 
#endif

/* Define to 1 if you have the <sys/types.h> header file. */
#ifndef __GIVARO_HAVE_SYS_TYPES_H 
#define __GIVARO_HAVE_SYS_TYPES_H  1 
#endif

/* Define to 1 if you have the <unistd.h> header file. */
#ifndef __GIVARO_HAVE_UNISTD_H 
#define __GIVARO_HAVE_UNISTD_H  1 
#endif

/* Define if you want most code inlined */
/* #undef __GIVARO_INLINE_ALL */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#ifndef __GIVARO_LT_OBJDIR 
#define __GIVARO_LT_OBJDIR  ".libs/" 
#endif

/* Name of package */
#ifndef __GIVARO_PACKAGE 
#define __GIVARO_PACKAGE  "givaro" 
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef __GIVARO_PACKAGE_BUGREPORT 
#define __GIVARO_PACKAGE_BUGREPORT  "http://github.com/linbox-team/givaro" 
#endif

/* Define to the full name of this package. */
#ifndef __GIVARO_PACKAGE_NAME 
#define __GIVARO_PACKAGE_NAME  "Givaro" 
#endif

/* Define to the full name and version of this package. */
#ifndef __GIVARO_PACKAGE_STRING 
#define __GIVARO_PACKAGE_STRING  "Givaro 4.2.0" 
#endif

/* Define to the one symbol short name of this package. */
#ifndef __GIVARO_PACKAGE_TARNAME 
#define __GIVARO_PACKAGE_TARNAME  "givaro" 
#endif

/* Define to the home page for this package. */
#ifndef __GIVARO_PACKAGE_URL 
#define __GIVARO_PACKAGE_URL  "https://casys.gricad-pages.univ-grenoble-alpes.fr/givaro" 
#endif

/* Define to the version of this package. */
#ifndef __GIVARO_PACKAGE_VERSION 
#define __GIVARO_PACKAGE_VERSION  "4.2.0" 
#endif

/* The size of `char', as computed by sizeof. */
#ifndef __GIVARO_SIZEOF_CHAR 
#define __GIVARO_SIZEOF_CHAR  1 
#endif

/* The size of `int', as computed by sizeof. */
#ifndef __GIVARO_SIZEOF_INT 
#define __GIVARO_SIZEOF_INT  4 
#endif

/* The size of `long', as computed by sizeof. */
#ifndef __GIVARO_SIZEOF_LONG 
#define __GIVARO_SIZEOF_LONG  8 
#endif

/* The size of `long long', as computed by sizeof. */
#ifndef __GIVARO_SIZEOF_LONG_LONG 
#define __GIVARO_SIZEOF_LONG_LONG  8 
#endif

/* The size of `short', as computed by sizeof. */
#ifndef __GIVARO_SIZEOF_SHORT 
#define __GIVARO_SIZEOF_SHORT  2 
#endif

/* The size of `__int64_t', as computed by sizeof. */
#ifndef __GIVARO_SIZEOF___INT64_T 
#define __GIVARO_SIZEOF___INT64_T  8 
#endif

/* Define to 1 if you have the ANSI C header files. */
#ifndef __GIVARO_STDC_HEADERS 
#define __GIVARO_STDC_HEADERS  1 
#endif

/* Version number of package */
#ifndef __GIVARO_VERSION 
#define __GIVARO_VERSION  "4.2.0" 
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
 
/* once: _GIVARO_CONFIG_H */
#endif
