// A single-threaded proactor for IREE's generic platform on ESP32-P4.
//
// IREE 3.13's local-sync device needs a proactor for its notifications
// (pool and semaphore wakeups), and iree_async_proactor_create_platform has
// none for IREE_PLATFORM_GENERIC (it returns UNAVAILABLE). This file defines
// that function; the runtime archive's version sits alone in its object file,
// so the linker takes this one and never pulls the other.
//
// Everything runs on one thread and local-sync executes work inline, so a
// notification never has to block: signal() only advances the epoch (done by
// iree_async_notification_signal before it calls us), and wait() reports
// whether the epoch has moved past the caller's token. Async operations,
// sockets, files, events, relays and fences are unimplemented. Threads too:
// see iree_thread_create below.

#include <string.h>

#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/base/threading/thread.h"

#define UNIMPLEMENTED(what) \
  iree_make_status(IREE_STATUS_UNIMPLEMENTED, "p4 proactor: " what)

static void p4_destroy(iree_async_proactor_t* proactor) {
  iree_allocator_free(proactor->allocator, proactor);
}

static iree_async_proactor_capabilities_t p4_query_capabilities(
    iree_async_proactor_t* proactor) {
  (void)proactor;
  return 0;
}

static iree_status_t p4_submit(iree_async_proactor_t* proactor,
                               iree_async_operation_list_t operations) {
  (void)proactor;
  (void)operations;
  return UNIMPLEMENTED("async operations");
}

static iree_status_t p4_poll(iree_async_proactor_t* proactor,
                             iree_timeout_t timeout,
                             iree_host_size_t* out_completed_count) {
  (void)proactor;
  (void)timeout;
  if (out_completed_count) *out_completed_count = 0;
  return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}

static void p4_wake(iree_async_proactor_t* proactor) { (void)proactor; }

static iree_status_t p4_cancel(iree_async_proactor_t* proactor,
                               iree_async_operation_t* operation) {
  (void)proactor;
  (void)operation;
  return UNIMPLEMENTED("cancel");
}

static iree_status_t p4_create_socket(iree_async_proactor_t* proactor,
                                      iree_async_socket_type_t type,
                                      iree_async_socket_options_t options,
                                      iree_async_socket_t** out_socket) {
  (void)proactor;
  (void)type;
  (void)options;
  *out_socket = NULL;
  return UNIMPLEMENTED("sockets");
}

static iree_status_t p4_import_socket(iree_async_proactor_t* proactor,
                                      iree_async_primitive_t primitive,
                                      iree_async_socket_type_t type,
                                      iree_async_socket_flags_t flags,
                                      iree_async_socket_t** out_socket) {
  (void)proactor;
  (void)primitive;
  (void)type;
  (void)flags;
  *out_socket = NULL;
  return UNIMPLEMENTED("sockets");
}

static void p4_destroy_socket(iree_async_proactor_t* proactor,
                              iree_async_socket_t* socket) {
  (void)proactor;
  (void)socket;
}

static iree_status_t p4_import_file(iree_async_proactor_t* proactor,
                                    iree_async_primitive_t primitive,
                                    iree_async_file_t** out_file) {
  (void)proactor;
  (void)primitive;
  *out_file = NULL;
  return UNIMPLEMENTED("files");
}

static void p4_destroy_file(iree_async_proactor_t* proactor,
                            iree_async_file_t* file) {
  (void)proactor;
  (void)file;
}

static iree_status_t p4_create_event(iree_async_proactor_t* proactor,
                                     iree_async_event_t** out_event) {
  (void)proactor;
  *out_event = NULL;
  return UNIMPLEMENTED("events");
}

static void p4_destroy_event(iree_async_proactor_t* proactor,
                             iree_async_event_t* event) {
  (void)proactor;
  (void)event;
}

static iree_status_t p4_register_event_source(
    iree_async_proactor_t* proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  (void)proactor;
  (void)handle;
  (void)callback;
  *out_event_source = NULL;
  return UNIMPLEMENTED("event sources");
}

static void p4_unregister_event_source(iree_async_proactor_t* proactor,
                                       iree_async_event_source_t* source) {
  (void)proactor;
  (void)source;
}

static iree_status_t p4_create_notification(
    iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  *out_notification = NULL;
  if (flags != IREE_ASYNC_NOTIFICATION_FLAG_NONE) {
    return UNIMPLEMENTED("notification flags");
  }
  iree_async_notification_t* notification = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      proactor->allocator, sizeof(*notification), (void**)&notification));
  memset(notification, 0, sizeof(*notification));
  iree_atomic_ref_count_init(&notification->ref_count);
  notification->proactor = proactor;
  iree_atomic_store(&notification->epoch, 0, iree_memory_order_release);
  notification->epoch_ptr = &notification->epoch;
  notification->flags = flags;
  notification->mode = IREE_ASYNC_NOTIFICATION_MODE_EVENT;
  *out_notification = notification;
  return iree_ok_status();
}

static iree_status_t p4_create_notification_shared(
    iree_async_proactor_t* proactor,
    const iree_async_notification_shared_options_t* options,
    iree_async_notification_t** out_notification) {
  (void)proactor;
  (void)options;
  *out_notification = NULL;
  return UNIMPLEMENTED("shared notifications");
}

static void p4_destroy_notification(iree_async_proactor_t* proactor,
                                    iree_async_notification_t* notification) {
  iree_allocator_free(proactor->allocator, notification);
}

static void p4_notification_signal(iree_async_proactor_t* proactor,
                                   iree_async_notification_t* notification,
                                   int32_t wake_count) {
  // The epoch is already advanced; there is no other thread to wake.
  (void)proactor;
  (void)notification;
  (void)wake_count;
}

static bool p4_notification_wait(iree_async_proactor_t* proactor,
                                 iree_async_notification_t* notification,
                                 uint32_t wait_token, iree_timeout_t timeout) {
  // Nothing else runs while we wait, so waiting longer cannot change the
  // answer: report whether a signal already happened.
  (void)proactor;
  (void)timeout;
  return (uint32_t)iree_atomic_load(notification->epoch_ptr,
                                    iree_memory_order_acquire) != wait_token;
}

static iree_status_t p4_register_relay(
    iree_async_proactor_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  (void)proactor;
  (void)source;
  (void)sink;
  (void)flags;
  (void)error_callback;
  *out_relay = NULL;
  return UNIMPLEMENTED("relays");
}

static void p4_unregister_relay(iree_async_proactor_t* proactor,
                                iree_async_relay_t* relay) {
  (void)proactor;
  (void)relay;
}

static iree_status_t p4_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)proactor;
  (void)state;
  (void)buffer;
  (void)access_flags;
  *out_entry = NULL;
  return UNIMPLEMENTED("buffer registration");
}

static iree_status_t p4_register_dmabuf(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)proactor;
  (void)state;
  (void)dmabuf_fd;
  (void)offset;
  (void)length;
  (void)access_flags;
  *out_entry = NULL;
  return UNIMPLEMENTED("dmabuf");
}

static void p4_unregister_buffer(iree_async_proactor_t* proactor,
                                 iree_async_buffer_registration_entry_t* entry,
                                 iree_async_buffer_registration_state_t* state) {
  (void)proactor;
  (void)entry;
  (void)state;
}

static iree_status_t p4_register_slab(iree_async_proactor_t* proactor,
                                      iree_async_slab_t* slab,
                                      iree_async_buffer_access_flags_t access_flags,
                                      iree_async_region_t** out_region) {
  (void)proactor;
  (void)slab;
  (void)access_flags;
  *out_region = NULL;
  return UNIMPLEMENTED("slabs");
}

static iree_status_t p4_import_fence(iree_async_proactor_t* proactor,
                                     iree_async_primitive_t fence,
                                     iree_async_semaphore_t* semaphore,
                                     uint64_t signal_value) {
  (void)proactor;
  (void)fence;
  (void)semaphore;
  (void)signal_value;
  return UNIMPLEMENTED("fences");
}

static iree_status_t p4_export_fence(iree_async_proactor_t* proactor,
                                     iree_async_semaphore_t* semaphore,
                                     uint64_t wait_value,
                                     iree_async_primitive_t* out_fence) {
  (void)proactor;
  (void)semaphore;
  (void)wait_value;
  (void)out_fence;
  return UNIMPLEMENTED("fences");
}

static const iree_async_proactor_vtable_t p4_vtable = {
    .destroy = p4_destroy,
    .query_capabilities = p4_query_capabilities,
    .submit = p4_submit,
    .poll = p4_poll,
    .wake = p4_wake,
    .cancel = p4_cancel,
    .create_socket = p4_create_socket,
    .import_socket = p4_import_socket,
    .destroy_socket = p4_destroy_socket,
    .import_file = p4_import_file,
    .destroy_file = p4_destroy_file,
    .create_event = p4_create_event,
    .destroy_event = p4_destroy_event,
    .register_event_source = p4_register_event_source,
    .unregister_event_source = p4_unregister_event_source,
    .create_notification = p4_create_notification,
    .create_notification_shared = p4_create_notification_shared,
    .destroy_notification = p4_destroy_notification,
    .notification_signal = p4_notification_signal,
    .notification_wait = p4_notification_wait,
    .register_relay = p4_register_relay,
    .unregister_relay = p4_unregister_relay,
    .register_buffer = p4_register_buffer,
    .register_dmabuf = p4_register_dmabuf,
    .unregister_buffer = p4_unregister_buffer,
    .register_slab = p4_register_slab,
    .import_fence = p4_import_fence,
    .export_fence = p4_export_fence,
    // Messaging and signals: NULL means unsupported.
};

iree_status_t iree_async_proactor_create_platform(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  (void)options;
  *out_proactor = NULL;
  iree_async_proactor_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*proactor), (void**)&proactor));
  iree_async_proactor_initialize(&p4_vtable, iree_make_cstring_view("p4"),
                                 allocator, proactor);
  *out_proactor = proactor;
  return iree_ok_status();
}

// The proactor pool's poll-thread runner links against these; the firmware
// creates its pool without a runner, so they are never called. The runtime
// is built without threading (IREE_ENABLE_THREADING=OFF).
iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread) {
  (void)entry;
  (void)entry_arg;
  (void)params;
  (void)allocator;
  *out_thread = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "p4: no threads");
}

void iree_thread_release(iree_thread_t* thread) { (void)thread; }
