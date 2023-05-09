/*******************************************************************************
 * Copyright (c) 2021 Tampere University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#ifndef __CL_EXT_POCL_H
#define __CL_EXT_POCL_H

#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifdef __cplusplus
extern "C"
{
#endif

/***********************************
* cl_pocl_content_size extension   *
************************************/

#define cl_pocl_content_size 1

extern CL_API_ENTRY cl_int CL_API_CALL
clSetContentSizeBufferPoCL(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clSetContentSizeBufferPoCL_fn)(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;


/***********************************
* cl_pocl_svm_rect +
* cl_pocl_command_buffer_svm +
* cl_pocl_command_buffer_host_exec +
* cl_pocl_command_buffer_host_buffer
* extensions
************************************/

// SVM copy/fill functions
#define cl_pocl_command_buffer_svm 1

// cl_mem & host related functions (clCommandReadBuffer etc)
#define cl_pocl_command_buffer_host_buffer 1

// clCommandHostFuncPOCL, clCommandWaitForEventPOCL, clCommandSignalEventPOCL
#define cl_pocl_command_buffer_host_exec 1

// clEnqueueSVMMemFillRectPOCL, clEnqueueSVMMemcpyRectPOCL
#define cl_pocl_svm_rect 1

/****************************************************/

/* cl_device_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_CAPABILITY_PROFILING_POCL  (1 << 8)

/* cl_command_buffer_flags_khr */
#define CL_COMMAND_BUFFER_PROFILING_POCL              (1 << 8)

/* cl_command_buffer_info_khr */
#define CL_COMMAND_BUFFER_INFO_PROFILING_POCL                     0x1299

/* cl_command_type */
/* To be used by clGetEventInfo: */
/* TODO use values from an assigned range */
#define CL_COMMAND_SVM_MEMCPY_RECT_POCL                       0x1210
#define CL_COMMAND_SVM_MEMFILL_RECT_POCL                      0x1211


typedef cl_int (CL_API_CALL *
clCommandSVMMemcpyPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSVMMemcpyRectPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    const size_t *dst_origin,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSVMMemfillPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    size_t size,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);


typedef cl_int (CL_API_CALL *
clCommandSVMMemfillRectPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    const size_t *origin,
    const size_t *region,
    size_t row_pitch,
    size_t slice_pitch,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);



typedef void (*CmdBufferCallbackFn_t)(void* userData);

typedef cl_int (CL_API_CALL *
clCommandHostFuncPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    CmdBufferCallbackFn_t callback_fn,
    void* user_data,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWaitForEventPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_event Event,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSignalEventPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_event *Event, // output
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);



typedef cl_int (CL_API_CALL *
clCommandReadBufferPOCL_fn)(cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem buffer,
                        size_t offset,
                        size_t size,
                        void *ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandReadBufferRectPOCL_fn)(cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_mem buffer,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch,
                            void *ptr,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr* sync_point_wait_list,
                            cl_sync_point_khr* sync_point,
                            cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandReadImagePOCL_fn)(cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_mem               image,
                       const size_t *       origin, /* [3] */
                       const size_t *       region, /* [3] */
                       size_t               row_pitch,
                       size_t               slice_pitch,
                       void *               ptr,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point,
                       cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteBufferPOCL_fn)(cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         const void *ptr,
                         cl_uint num_sync_points_in_wait_list,
                         const cl_sync_point_khr* sync_point_wait_list,
                         cl_sync_point_khr* sync_point,
                         cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteBufferRectPOCL_fn)(cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             cl_mem buffer,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch,
                             const void *ptr,
                             cl_uint num_sync_points_in_wait_list,
                             const cl_sync_point_khr* sync_point_wait_list,
                             cl_sync_point_khr* sync_point,
                             cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteImagePOCL_fn)(cl_command_buffer_khr command_buffer,
                        cl_command_queue    command_queue,
                        cl_mem              image,
                        const size_t *      origin, /*[3]*/
                        const size_t *      region, /*[3]*/
                        size_t              row_pitch,
                        size_t              slice_pitch,
                        const void *        ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clEnqueueSVMMemcpyRectPOCL_fn) (cl_command_queue command_queue,
                            cl_bool blocking,
                            void *dst_ptr,
                            const void *src_ptr,
                            const size_t *dst_origin,
                            const size_t *src_origin,
                            const size_t *region,
                            size_t dst_row_pitch,
                            size_t dst_slice_pitch,
                            size_t src_row_pitch,
                            size_t src_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

typedef cl_int (CL_API_CALL *
clEnqueueSVMMemFillRectPOCL_fn) (cl_command_queue  command_queue,
                             void *            svm_ptr,
                             const size_t *    origin,
                             const size_t *    region,
                             size_t            row_pitch,
                             size_t            slice_pitch,
                             const void *      pattern,
                             size_t            pattern_size,
                             size_t            size,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event);


#ifndef CL_NO_PROTOTYPES

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyRectPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    const size_t *dst_origin,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemfillPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    size_t size,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemfillRectPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    const size_t *origin,
    const size_t *region,
    size_t row_pitch,
    size_t slice_pitch,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);




extern CL_API_ENTRY cl_int CL_API_CALL
clCommandHostFuncPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    CmdBufferCallbackFn_t callback_fn,
    void* user_data,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWaitForEventPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_event Event,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSignalEventPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_event *Event, // output
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);


extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadBufferPOCL(cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem buffer,
                        size_t offset,
                        size_t size,
                        void *ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadBufferRectPOCL(cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_mem buffer,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch,
                            void *ptr,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr* sync_point_wait_list,
                            cl_sync_point_khr* sync_point,
                            cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadImagePOCL(cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_mem               image,
                       const size_t *       origin, /* [3] */
                       const size_t *       region, /* [3] */
                       size_t               row_pitch,
                       size_t               slice_pitch,
                       void *               ptr,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point,
                       cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteBufferPOCL(cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         const void *ptr,
                         cl_uint num_sync_points_in_wait_list,
                         const cl_sync_point_khr* sync_point_wait_list,
                         cl_sync_point_khr* sync_point,
                         cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteBufferRectPOCL(cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             cl_mem buffer,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch,
                             const void *ptr,
                             cl_uint num_sync_points_in_wait_list,
                             const cl_sync_point_khr* sync_point_wait_list,
                             cl_sync_point_khr* sync_point,
                             cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteImagePOCL(cl_command_buffer_khr command_buffer,
                        cl_command_queue    command_queue,
                        cl_mem              image,
                        const size_t *      origin, /*[3]*/
                        const size_t *      region, /*[3]*/
                        size_t              row_pitch,
                        size_t              slice_pitch,
                        const void *        ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpyRectPOCL (cl_command_queue command_queue,
                            cl_bool blocking,
                            void *dst_ptr,
                            const void *src_ptr,
                            const size_t *dst_origin,
                            const size_t *src_origin,
                            const size_t *region,
                            size_t dst_row_pitch,
                            size_t dst_slice_pitch,
                            size_t src_row_pitch,
                            size_t src_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFillRectPOCL (cl_command_queue  command_queue,
                             void *            svm_ptr,
                             const size_t *    origin,
                             const size_t *    region,
                             size_t            row_pitch,
                             size_t            slice_pitch,
                             const void *      pattern,
                             size_t            pattern_size,
                             size_t            size,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event);


#endif

#ifdef __cplusplus
}
#endif

#endif /* __CL_EXT_POCL_H */
