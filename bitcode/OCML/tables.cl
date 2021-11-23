/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// #include "ocml.h"

#include "tables.h"


#define DECLARE_TABLE(TYPE,NAME,LENGTH) \
__attribute__((visibility("protected"))) __constant TYPE TABLE_MANGLE(NAME) [ LENGTH ] = {

#define END_TABLE() };

#include "besselF_table.h"
#include "besselD_table.h"


