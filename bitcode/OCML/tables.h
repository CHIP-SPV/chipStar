/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// Table stuff

#define TABLE_MANGLE(NAME) __ocmltbl_##NAME

extern __constant float TABLE_MANGLE(M32_J0)[];
extern __constant float TABLE_MANGLE(M32_J1)[];
extern __constant float TABLE_MANGLE(M32_Y0)[];
extern __constant float TABLE_MANGLE(M32_Y1)[];
extern __constant double TABLE_MANGLE(M64_J0)[];
extern __constant double TABLE_MANGLE(M64_J1)[];
extern __constant double TABLE_MANGLE(M64_Y0)[];
extern __constant double TABLE_MANGLE(M64_Y1)[];

#define USE_TABLE(TYPE,PTR,NAME) \
    __constant TYPE * PTR = TABLE_MANGLE(NAME)

