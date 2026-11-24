// Runs a BioCLIP 2 image encoder on an iPhone and records what compare.py
// needs to score it on the Mac.
//
// The bundle's Model/ folder carries pixels.f16 (N x 3 x 224 x 224 f16, the
// reference photos from export.py) and either model.vmfb (IREE on Metal) or
// model.mlmodelc (Core ML, compute units ALL, function b1). On launch the
// app:
//   1. prints the GPU's Metal architecture name (MLX picks tiles from it);
//   2. encodes every photo once and writes the outputs to
//      Documents/embeddings.bin;
//   3. times `kTimedRuns` calls on the first photo after `kWarmupRuns`;
//   4. writes Documents/result.json and exits.
// Lines prefixed "BENCH " go to stdout for `devicectl ... --console`.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#include <mach/mach_time.h>

#include "iree/runtime/api.h"

static const int kWarmupRuns = 5;
static const int kTimedRuns = 30;
static const iree_hal_dim_t kImageShape[4] = {1, 3, 224, 224};
static const NSUInteger kImageBytes = 3 * 224 * 224 * sizeof(uint16_t);

// Encodes one image (f16 NCHW) and returns the embedding's bytes.
typedef NSData* (^EncodeBlock)(const uint16_t* image);

// What a backend reports besides its encoder.
@interface Backend : NSObject
@property(copy, nonatomic) NSString* name;
@property(copy, nonatomic) EncodeBlock encode;
@property(nonatomic) uint32_t outputElementType;  // IREE HAL element type
@property(copy, nonatomic) void (^cleanup)(void);
@end
@implementation Backend
@end

static NSString* thermal_state(void) {
  switch (NSProcessInfo.processInfo.thermalState) {
    case NSProcessInfoThermalStateNominal: return @"nominal";
    case NSProcessInfoThermalStateFair: return @"fair";
    case NSProcessInfoThermalStateSerious: return @"serious";
    case NSProcessInfoThermalStateCritical: return @"critical";
  }
  return @"unknown";
}

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

static Backend* load_iree(NSString* vmfb) {
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

  Backend* backend = [Backend new];
  backend.name = @"iree-metal";
  backend.outputElementType = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  backend.encode = ^NSData*(const uint16_t* image) {
    iree_hal_buffer_view_t* out = encode(session, image, kImageBytes);
    NSData* bytes = read_back(session, out);
    if (iree_hal_buffer_view_element_type(out) != IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
      printf("BENCH error: output is not f16\n");
      exit(1);
    }
    iree_hal_buffer_view_release(out);
    return bytes;
  };
  backend.cleanup = ^{
    iree_runtime_session_release(session);
    iree_runtime_instance_release(instance);
  };
  return backend;
}

static Backend* load_coreml(NSString* mlmodelc) {
  MLModelConfiguration* config = [MLModelConfiguration new];
  config.computeUnits = MLComputeUnitsAll;
  config.functionName = @"b1";
  NSError* error = nil;
  MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:mlmodelc]
                                     configuration:config
                                             error:&error];
  if (!model) {
    printf("BENCH error loading Core ML model: %s\n", error.description.UTF8String);
    exit(1);
  }
  // The model takes f32 pixels; one reusable input array.
  MLMultiArray* input = [[MLMultiArray alloc] initWithShape:@[ @1, @3, @224, @224 ]
                                                   dataType:MLMultiArrayDataTypeFloat32
                                                      error:&error];
  Backend* backend = [Backend new];
  backend.name = @"coreml-all";
  backend.outputElementType = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  backend.encode = ^NSData*(const uint16_t* image) {
    float* dst = (float*)input.dataPointer;
    const _Float16* src = (const _Float16*)image;
    for (NSUInteger i = 0; i < kImageBytes / 2; ++i) dst[i] = (float)src[i];
    NSError* err = nil;
    id<MLFeatureProvider> features =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"pixel_values" : input}
                                                          error:&err];
    id<MLFeatureProvider> out = [model predictionFromFeatures:features error:&err];
    if (!out) {
      printf("BENCH error: %s\n", err.description.UTF8String);
      exit(1);
    }
    MLMultiArray* emb = [out featureValueForName:@"image_features"].multiArrayValue;
    NSMutableData* bytes = [NSMutableData dataWithLength:emb.count * sizeof(uint16_t)];
    _Float16* o = (_Float16*)bytes.mutableBytes;
    if (emb.dataType == MLMultiArrayDataTypeFloat16) {
      [emb getBytesWithHandler:^(const void* p, NSInteger size) {
        memcpy(o, p, MIN((NSUInteger)size, bytes.length));
      }];
    } else {
      for (NSInteger i = 0; i < emb.count; ++i) o[i] = (_Float16)emb[i].floatValue;
    }
    return bytes;
  };
  backend.cleanup = ^{
  };
  return backend;
}

static void run(void) {
  NSBundle* bundle = NSBundle.mainBundle;
  NSString* vmfb = [bundle pathForResource:@"model" ofType:@"vmfb" inDirectory:@"Model"];
  NSString* mlmodelc = [bundle pathForResource:@"model" ofType:@"mlmodelc" inDirectory:@"Model"];
  NSData* pixels =
      [NSData dataWithContentsOfFile:[bundle pathForResource:@"pixels" ofType:@"f16" inDirectory:@"Model"]
                             options:NSDataReadingMappedIfSafe
                               error:nil];
  const int images = (int)(pixels.length / kImageBytes);

  id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
  NSString* arch = gpu.architecture.name;
  NSString* thermal_start = thermal_state();
  printf("BENCH gpu %s (%s), %d images, thermal state %s\n", gpu.name.UTF8String, arch.UTF8String,
         images, thermal_start.UTF8String);

  double t0 = now_ms();
  Backend* backend = vmfb ? load_iree(vmfb) : load_coreml(mlmodelc);
  double load_ms = now_ms() - t0;
  printf("BENCH runtime %s\n", backend.name.UTF8String);
  const uint16_t* first = (const uint16_t*)pixels.bytes;

  // Every photo once, for accuracy.
  NSMutableData* embeddings = [NSMutableData data];
  NSMutableArray* first_call_ms = [NSMutableArray array];
  NSUInteger out_count = 0;
  for (int i = 0; i < images; ++i) {
    double t = now_ms();
    NSData* bytes = backend.encode(first + i * kImageBytes / 2);
    [first_call_ms addObject:@(now_ms() - t)];
    [embeddings appendData:bytes];
    out_count = bytes.length / sizeof(uint16_t);
  }

  // Latency on the first photo: call plus read-back, as compare.py times it.
  for (int i = 0; i < kWarmupRuns; ++i) (void)backend.encode(first);
  NSMutableArray* times_ms = [NSMutableArray array];
  for (int i = 0; i < kTimedRuns; ++i) {
    double t = now_ms();
    (void)backend.encode(first);
    [times_ms addObject:@(now_ms() - t)];
  }
  NSArray* sorted = [times_ms sortedArrayUsingSelector:@selector(compare:)];
  double median = [sorted[kTimedRuns / 2] doubleValue];
  NSString* thermal_end = thermal_state();
  printf("BENCH load %.1f ms, call median %.2f ms (min %.2f, max %.2f) over %d runs, thermal state %s\n",
         load_ms, median, [sorted.firstObject doubleValue], [sorted.lastObject doubleValue], kTimedRuns,
         thermal_end.UTF8String);

  NSDictionary* result = @{
    @"runtime" : backend.name,
    @"gpu" : gpu.name,
    @"arch" : arch,
    @"os" : UIDevice.currentDevice.systemVersion,
    @"images" : @(images),
    @"output_element_type" : @(backend.outputElementType),
    @"output_shape" : @[ @1, @(out_count) ],
    @"load_ms" : @(load_ms),
    @"first_call_ms" : first_call_ms,
    @"times_ms" : times_ms,
    @"thermal_state" : @[ thermal_start, thermal_end ],
  };
  NSString* docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
  [embeddings writeToFile:[docs stringByAppendingPathComponent:@"embeddings.bin"] atomically:YES];
  [[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil]
      writeToFile:[docs stringByAppendingPathComponent:@"result.json"]
       atomically:YES];

  backend.cleanup();
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
