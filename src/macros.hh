#ifndef TEMP_H
#define TEMP_H

#define RETURN(x)                  \
  do {                             \
    hipError_t err = (x);          \
    Backend->tls_last_error = err; \
    return err;                    \
  } while (0)

#define ERROR_IF(cond, err)                                                  \
  if (cond) do {                                                             \
      logError("Error {} at {}:{} code {}", err, __FILE__, __LINE__, #cond); \
      Backend->tls_last_error = err;                                         \
      return err;                                                            \
  } while (0)

#endif