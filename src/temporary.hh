#ifndef TEMP_H
#define TEMP_H

//#define dim3 int
// #define logDebug(x, y) printf(x, y)
// #define logError(x, y) printf(x, y)
// #define logCritical(x, y) printf(x, y)

//    tls_LastError = err;
#define RETURN(x)         \
  do {                    \
    hipError_t err = (x); \
    return err;           \
  } while (0)

// tls_LastError = err;
#define ERROR_IF(cond, err)                                                  \
  if (cond) do {                                                             \
      logError("Error {} at {}:{} code {}", err, __FILE__, __LINE__, #cond); \
      return err;                                                            \
  } while (0)

#endif