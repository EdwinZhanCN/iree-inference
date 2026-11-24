// Runs an IREE Metal BioCLIP 2 encoder on an iPhone and records what
// compare.py needs to score it on the Mac.
//
// The bundle carries model.vmfb and pixels.f16 (N x 3 x 224 x 224 f16, the
// reference photos from export.py). On launch the app:
//   1. prints the GPU's Metal architecture name (MLX picks tiles from it);
//   2. encodes every photo once and writes the outputs to
//      Documents/embeddings.bin;
//   3. times `kTimedRuns` calls on the first photo after `kWarmupRuns`;
//   4. writes Documents/result.json and exits.
// Lines prefixed "BENCH " go to stdout for `devicectl ... --console`.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#include <mach/mach_time.h>

#include "iree/runtime/api.h"

static const int kWarmupRuns = 5;
static const int kTimedRuns = 30;
static const iree_hal_dim_t kImageShape[4] = {1, 3, 224, 224};

static double now_ms(void) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static NSString* status_string(iree_status_t status) {
  char* buffer = NULL;
  iree_host_size_t length = 0;
  NSString* text = @"(unprintable status)";
  iree_allocator_t allocator = iree_allocator_system();
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    text = [[NSString alloc] initWithBytes:buffer length:length encoding:NSUTF8StringEncoding];
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return text;
}

#define CHECK(expr)                                                        \
  do {                                                                     \
    iree_status_t s_ = (expr);                                             \
    if (!iree_status_is_ok(s_)) {                                          \
      printf("BENCH error %s: %s\n", #expr, status_string(s_).UTF8String); \
      fflush(stdout);                                                      \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// Calls module.main on one image; returns the output buffer view.
static iree_hal_buffer_view_t* encode(iree_runtime_session_t* session, const uint16_t* image,
                                      iree_host_size_t image_bytes) {
  iree_runtime_call_t call;
  CHECK(iree_runtime_call_initialize_by_name(session, iree_make_cstring_view("module.main"), &call));
  iree_hal_buffer_view_t* input = NULL;
  CHECK(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(session), iree_runtime_session_device_allocator(session), 4,
      kImageShape, IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(image, image_bytes), &input));
  CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, input));
  iree_hal_buffer_view_release(input);
  CHECK(iree_runtime_call_invoke(&call, 0));
  iree_hal_buffer_view_t* output = NULL;
  CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &output));
  iree_runtime_call_deinitialize(&call);
  return output;
}

static NSData* read_back(iree_runtime_session_t* session, iree_hal_buffer_view_t* view) {
  iree_device_size_t length = iree_hal_buffer_view_byte_length(view);
  NSMutableData* data = [NSMutableData dataWithLength:length];
  CHECK(iree_hal_device_transfer_d2h(iree_runtime_session_device(session),
                                     iree_hal_buffer_view_buffer(view), 0, data.mutableBytes,
                                     length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                     iree_infinite_timeout()));
  return data;
}

static void run(void) {
  NSBundle* bundle = NSBundle.mainBundle;
  NSString* vmfb = [bundle pathForResource:@"model" ofType:@"vmfb" inDirectory:@"Model"];
  NSData* pixels = [NSData dataWithContentsOfFile:[bundle pathForResource:@"pixels" ofType:@"f16" inDirectory:@"Model"]
                                          options:NSDataReadingMappedIfSafe
                                            error:nil];
  const iree_host_size_t image_bytes = 3 * 224 * 224 * sizeof(uint16_t);
  const int images = (int)(pixels.length / image_bytes);

  id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
  NSString* arch = gpu.architecture.name;
  printf("BENCH gpu %s (%s), %d images\n", gpu.name.UTF8String, arch.UTF8String, images);

  double t0 = now_ms();
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  CHECK(iree_runtime_instance_create(&instance_options, iree_allocator_system(), &instance));
  iree_hal_device_t* device = NULL;
  CHECK(iree_runtime_instance_try_create_default_device(instance, iree_make_cstring_view("metal"),
                                                        &device));
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  CHECK(iree_runtime_session_create_with_device(instance, &session_options, device,
                                                iree_runtime_instance_host_allocator(instance),
                                                &session));
  iree_hal_device_release(device);
  CHECK(iree_runtime_session_append_bytecode_module_from_file(session, vmfb.fileSystemRepresentation));
  double load_ms = now_ms() - t0;

  // Every photo once, for accuracy.
  NSMutableData* embeddings = [NSMutableData data];
  NSMutableArray* first_call_ms = [NSMutableArray array];
  iree_host_size_t out_rank = 0;
  iree_hal_dim_t out_shape[4] = {0};
  iree_hal_element_type_t out_type = IREE_HAL_ELEMENT_TYPE_NONE;
  for (int i = 0; i < images; ++i) {
    double t = now_ms();
    iree_hal_buffer_view_t* out =
        encode(session, (const uint16_t*)pixels.bytes + i * image_bytes / 2, image_bytes);
    NSData* bytes = read_back(session, out);
    [first_call_ms addObject:@(now_ms() - t)];
    [embeddings appendData:bytes];
    out_type = iree_hal_buffer_view_element_type(out);
    out_rank = iree_hal_buffer_view_shape_rank(out);
    for (iree_host_size_t d = 0; d < out_rank && d < 4; ++d) {
      out_shape[d] = iree_hal_buffer_view_shape_dim(out, d);
    }
    iree_hal_buffer_view_release(out);
  }

  // Latency on the first photo: call plus read-back, as compare.py times it.
  for (int i = 0; i < kWarmupRuns; ++i) {
    iree_hal_buffer_view_t* out = encode(session, pixels.bytes, image_bytes);
    (void)read_back(session, out);
    iree_hal_buffer_view_release(out);
  }
  NSMutableArray* times_ms = [NSMutableArray array];
  for (int i = 0; i < kTimedRuns; ++i) {
    double t = now_ms();
    iree_hal_buffer_view_t* out = encode(session, pixels.bytes, image_bytes);
    (void)read_back(session, out);
    [times_ms addObject:@(now_ms() - t)];
    iree_hal_buffer_view_release(out);
  }
  NSArray* sorted = [times_ms sortedArrayUsingSelector:@selector(compare:)];
  double median = [sorted[kTimedRuns / 2] doubleValue];
  printf("BENCH load %.1f ms, call median %.2f ms (min %.2f, max %.2f) over %d runs\n", load_ms,
         median, [sorted.firstObject doubleValue], [sorted.lastObject doubleValue], kTimedRuns);

  NSMutableArray* shape = [NSMutableArray array];
  for (iree_host_size_t d = 0; d < out_rank; ++d) [shape addObject:@(out_shape[d])];
  NSDictionary* result = @{
    @"gpu" : gpu.name,
    @"arch" : arch,
    @"os" : UIDevice.currentDevice.systemVersion,
    @"images" : @(images),
    @"output_element_type" : @(out_type),
    @"output_shape" : shape,
    @"load_ms" : @(load_ms),
    @"first_call_ms" : first_call_ms,
    @"times_ms" : times_ms,
  };
  NSString* docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
  [embeddings writeToFile:[docs stringByAppendingPathComponent:@"embeddings.bin"] atomically:YES];
  [[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil]
      writeToFile:[docs stringByAppendingPathComponent:@"result.json"]
       atomically:YES];

  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  printf("BENCH done\n");
  fflush(stdout);
  exit(0);
}

// iOS 27 requires the scene lifecycle; the scene only shows a blank window.
@interface SceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

@implementation SceneDelegate
- (void)scene:(UIScene*)scene
    willConnectToSession:(UISceneSession*)session
                 options:(UISceneConnectionOptions*)options {
  self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
  self.window.rootViewController = [[UIViewController alloc] init];
  [self.window makeKeyAndVisible];
}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication*)application
    didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    run();
  });
  return YES;
}

- (UISceneConfiguration*)application:(UIApplication*)application
    configurationForConnectingSceneSession:(UISceneSession*)session
                                   options:(UISceneConnectionOptions*)options {
  UISceneConfiguration* config = [[UISceneConfiguration alloc] initWithName:@"Default"
                                                                sessionRole:session.role];
  config.delegateClass = [SceneDelegate class];
  return config;
}
@end

int main(int argc, char* argv[]) {
  // Unbuffered, so the console keeps what was printed if the app dies.
  setvbuf(stdout, NULL, _IONBF, 0);
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
  }
}
