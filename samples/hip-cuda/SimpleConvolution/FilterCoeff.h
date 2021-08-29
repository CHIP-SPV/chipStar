/*******************************************************************************
 Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1   Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 2   Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
/**
 ********************************************************************************
 * @file <filterCoeff.h>
 *
 * @brief Contains the filter coefficients.
 *
 ********************************************************************************
 */

#ifndef __SEPFILTERCOEFF__H
#define __SEPFILTERCOEFF__H

/********************************************************************************
 *
 * Sobel 3X3 filter:
 *                      1 0 -1
 *                      2 0 -2
 *                      1 0 -1
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {1.0f, 0.0f, -1.0f};
 * h = {1.0f, 2.0f, 1.0f};
 *********************************************************************************/
float SOBEL_FILTER_3x3[3*3] = 
							{
								 1.0f,  0.0f,  -1.0f,
								 2.0f,  0.0f,  -2.0f,
								 1.0f,  0.0f,  -1.0f
							};
float SOBEL_FILTER_3x3_pass1[3] = { 1.0f, 0.0f, -1.0f };
float SOBEL_FILTER_3x3_pass2[3] = { 1.0f, 2.0f, 1.0f };
/********************************************************************************
 *
 * Sobel 5X5 filter:
 *                       1 2 0 -2 -1; 
 *                       4 8 0 -8 -4;
 *                       6 12 0 -12 -6;
 *                       4 8 0 -8 -4;
 *                       1 2 0 -2 -1;
 *
 * This is a separable filter. The equivalent separable coefficients are:
 * v = {1.0f,  2.0f,   0.0f,  -2.0f,  -1.0f };
 * h = {1.0f,  4.0f,   6.0f,   4.0f,   1.0f };
 *********************************************************************************/
float SOBEL_FILTER_5x5[5*5] = 
							{
								 1.0f,  2.0f,   0.0f,  -2.0f,  -1.0f,
								 4.0f,  8.0f,   0.0f,  -8.0f,  -4.0f,
								 6.0f,  12.0f,  0.0f,  -12.0f, -6.0f,
								 4.0f,  8.0f,   0.0f,  -8.0f,  -4.0f,
								 1.0f,  2.0f,   0.0f,  -2.0f,  -1.0f
							};
float SOBEL_FILTER_5x5_pass1[5] = { 1.0f,  2.0f,   0.0f,  -2.0f,  -1.0f };
float SOBEL_FILTER_5x5_pass2[5] = { 1.0f,  4.0f,   6.0f,  4.0f,  1.0f };

#endif
