#import "VBSGraph.h"
#import "VBSGGUF.h"
#import <CommonCrypto/CommonDigest.h>

@implementation VBSNode @end
@implementation VBSGraph @end

static NSString *const GraphError = @"Verum.Graph";
static NSError *graphError(NSString *s){return [NSError errorWithDomain:GraphError code:1 userInfo:@{NSLocalizedDescriptionKey:s}];}

typedef NS_ENUM(NSUInteger, TensorRole) { RUnknown,REmbed,RNorm,RQuery,RKey,RValue,RAttentionOut,RGate,RUp,RDown,RConv,RState,RProjection };

// This vocabulary describes operations, not model families.  Shape constraints
// below must corroborate every name-derived role before it becomes executable.
static TensorRole roleForName(NSString *name) {
    NSArray<NSString*> *p=[[name lowercaseString] componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"._/"]];
    NSSet *s=[NSSet setWithArray:p];
    if([s containsObject:@"token"]||[s containsObject:@"tok"]||[s containsObject:@"embed"]||[s containsObject:@"embd"])return REmbed;
    if([s containsObject:@"norm"]||[s containsObject:@"ln"]||[s containsObject:@"rms"])return RNorm;
    if([s containsObject:@"query"]||[s containsObject:@"q"])return RQuery;
    if([s containsObject:@"key"]||[s containsObject:@"k"])return RKey;
    if([s containsObject:@"value"]||[s containsObject:@"v"])return RValue;
    if([s containsObject:@"gate"])return RGate;
    if([s containsObject:@"up"]||[s containsObject:@"expand"])return RUp;
    if([s containsObject:@"down"]||[s containsObject:@"contract"])return RDown;
    if([s containsObject:@"conv"]||[s containsObject:@"conv1d"])return RConv;
    if([s containsObject:@"ssm"]||[s containsObject:@"state"])return RState;
    if([s containsObject:@"output"]||[s containsObject:@"proj"]||[s containsObject:@"projection"])return RProjection;
    if([s containsObject:@"attention"]||[s containsObject:@"attn"])return RAttentionOut;
    return RUnknown;
}

static NSString *scopeForName(NSString *name) {
    NSArray *parts=[name componentsSeparatedByString:@"."]; NSMutableArray *scope=[NSMutableArray array];
    for(NSString *p in parts){NSScanner*sc=[NSScanner scannerWithString:p];NSInteger v;if([sc scanInteger:&v]&&sc.isAtEnd){[scope addObject:[NSString stringWithFormat:@"#%ld",(long)v]];break;}[scope addObject:p];}
    return [scope componentsJoinedByString:@"."];
}

static BOOL isMatrix(VBSTensor*t){return t.shape.count>=2&&t.columns>0&&t.rows>0;}
static BOOL compatible(VBSTensor*a,VBSTensor*b){return isMatrix(a)&&isMatrix(b)&&(a.columns==b.columns||a.rows==b.columns||a.columns==b.rows);}

@implementation VBSGraphCompiler
- (VBSGraph *)compileModel:(VBSGGUFModel *)model error:(NSError **)error {
    NSString *embedded=[model stringForMetadataKey:@"verum.graph.json"];
    if(embedded){NSData*d=[embedded dataUsingEncoding:NSUTF8StringEncoding];NSDictionary*j=[NSJSONSerialization JSONObjectWithData:d options:0 error:error];if(![j isKindOfClass:NSDictionary.class])return nil;return [self graphFromJSON:j model:model error:error];}

    NSMutableDictionary<NSString*,NSMutableArray<VBSTensor*>*>*scopes=[NSMutableDictionary dictionary];
    NSMutableArray<VBSTensor*>*unknown=[NSMutableArray array];
    for(VBSTensor*t in model.tensors){TensorRole r=roleForName(t.name);if(r==RUnknown){[unknown addObject:t];continue;}NSString*k=scopeForName(t.name);if(!scopes[k])scopes[k]=[NSMutableArray array];[scopes[k] addObject:t];}
    NSMutableArray<VBSNode*>*nodes=[NSMutableArray array];
    NSArray*ordered=[[scopes allKeys] sortedArrayUsingSelector:@selector(localizedStandardCompare:)];
    for(NSString*scope in ordered){NSArray<VBSTensor*>*ts=scopes[scope];NSMutableDictionary<NSNumber*,NSMutableArray<VBSTensor*>*>*roles=[NSMutableDictionary dictionary];for(VBSTensor*t in ts){NSNumber*k=@(roleForName(t.name));if(!roles[k])roles[k]=[NSMutableArray array];[roles[k] addObject:t];}
        NSArray*q=roles[@(RQuery)],*k=roles[@(RKey)],*v=roles[@(RValue)];
        if(q.count||k.count||v.count){if(q.count!=1||k.count!=1||v.count!=1||!compatible(q[0],k[0])||!compatible(q[0],v[0])){if(error)*error=graphError([@"Ambiguous attention tensors in " stringByAppendingString:scope]);return nil;}for(VBSTensor*w in @[q[0],k[0],v[0]])[nodes addObject:[self linear:w scope:scope]];VBSNode*a=[VBSNode new];a.identifier=[scope stringByAppendingString:@".attention"];a.kind=VBSOpAttention;a.inputs=@[[q[0] name],[k[0] name],[v[0] name]];a.outputs=@[[a.identifier stringByAppendingString:@".out"]];a.attributes=@{};[nodes addObject:a];}
        for(VBSTensor*t in ts){TensorRole r=roleForName(t.name);if(r==RNorm){VBSNode*n=[VBSNode new];n.identifier=t.name;n.kind=VBSOpNormalize;n.weight=t;n.inputs=@[];n.outputs=@[[t.name stringByAppendingString:@".out"]];n.attributes=@{};[nodes addObject:n];}else if(r==RConv||r==RState){VBSNode*n=[VBSNode new];n.identifier=t.name;n.kind=(r==RConv?VBSOpConvolution:VBSOpStateSpace);n.weight=t;n.inputs=@[];n.outputs=@[[t.name stringByAppendingString:@".out"]];n.attributes=@{};[nodes addObject:n];}else if(r==REmbed){VBSNode*n=[VBSNode new];n.identifier=t.name;n.kind=VBSOpEmbedding;n.weight=t;n.inputs=@[@"token"];n.outputs=@[[t.name stringByAppendingString:@".out"]];n.attributes=@{};[nodes addObject:n];}else if((r==RGate||r==RUp||r==RDown||r==RAttentionOut||r==RProjection)&&isMatrix(t)){[nodes addObject:[self linear:t scope:scope]];}}
    }
    if(nodes.count==0){if(error)*error=graphError(@"No executable semantics could be proven from GGUF metadata/tensors");return nil;}
    // Unknown matrices can change the graph.  Refuse them; unknown scalar
    // diagnostics do not block compilation.
    NSMutableArray*blocking=[NSMutableArray array];for(VBSTensor*t in unknown)if(isMatrix(t))[blocking addObject:t.name];
    if(blocking.count){if(error)*error=graphError([NSString stringWithFormat:@"Unresolved matrix semantics (%lu): %@",(unsigned long)blocking.count,[blocking componentsJoinedByString:@", "]]);return nil;}
    VBSGraph*g=[VBSGraph new];g.nodes=nodes;NSString*arch=[model stringForMetadataKey:@"general.architecture"]?:@"unspecified";g.facts=@{@"declaredArchitecture":arch,@"tensorCount":@(model.tensors.count),@"semanticSource":@"metadata+names+shapes"};g.fingerprint=[self fingerprint:nodes];return g;
}
- (VBSNode*)linear:(VBSTensor*)w scope:(NSString*)scope { VBSNode*n=[VBSNode new];n.identifier=w.name;n.kind=VBSOpLinear;n.weight=w;n.inputs=@[[scope stringByAppendingString:@".input"]];n.outputs=@[[w.name stringByAppendingString:@".out"]];n.attributes=@{@"columns":@(w.columns),@"rows":@(w.rows)};return n; }
- (NSString*)fingerprint:(NSArray<VBSNode*>*)nodes { NSMutableString*s=[NSMutableString string];for(VBSNode*n in nodes)[s appendFormat:@"%lu:%@:%@;",(unsigned long)n.kind,n.identifier,n.weight.shape?:@[]];unsigned char h[CC_SHA256_DIGEST_LENGTH];CC_SHA256(s.UTF8String,(CC_LONG)strlen(s.UTF8String),h);NSMutableString*out=[NSMutableString string];for(int i=0;i<CC_SHA256_DIGEST_LENGTH;i++)[out appendFormat:@"%02x",h[i]];return out; }
- (VBSGraph*)graphFromJSON:(NSDictionary*)json model:(VBSGGUFModel*)model error:(NSError**)error { NSArray*spec=json[@"nodes"];if(![spec isKindOfClass:NSArray.class]){if(error)*error=graphError(@"verum.graph.json has no nodes array");return nil;}NSMutableArray*out=[NSMutableArray array];for(NSDictionary*x in spec){if(![x isKindOfClass:NSDictionary.class]||![x[@"id"] isKindOfClass:NSString.class]||![x[@"op"] isKindOfClass:NSNumber.class]){if(error)*error=graphError(@"Invalid declarative graph node");return nil;}VBSNode*n=[VBSNode new];n.identifier=x[@"id"];NSUInteger _kind=[x[@"op"] unsignedIntegerValue];if(_kind>VBSOpProjection){if(error)*error=graphError([@"Unknown operation kind in verum.graph.json: " stringByAppendingString:n.identifier]);return nil;}n.kind=_kind;id _in=x[@"inputs"]?:@[],_out=x[@"outputs"]?:@[],_at=x[@"attributes"]?:@{};if(![_in isKindOfClass:NSArray.class]||![_out isKindOfClass:NSArray.class]||![_at isKindOfClass:NSDictionary.class]){if(error)*error=graphError([@"Malformed declarative graph node: " stringByAppendingString:n.identifier]);return nil;}n.inputs=_in;n.outputs=_out;n.attributes=_at;id _w=x[@"weight"];if(_w&&![_w isKindOfClass:NSString.class]){if(error)*error=graphError([@"Malformed weight reference in verum.graph.json: " stringByAppendingString:n.identifier]);return nil;}NSString*w=_w;if(w){n.weight=model.tensorsByName[w];if(!n.weight){if(error)*error=graphError([@"Missing declared weight: " stringByAppendingString:w]);return nil;}}if(n.kind==VBSOpLinear&&!n.weight){if(error)*error=graphError([@"Linear node declares no weight: " stringByAppendingString:n.identifier]);return nil;}[out addObject:n];}VBSGraph*g=[VBSGraph new];g.nodes=out;g.facts=@{@"semanticSource":@"verum.graph.json"};g.fingerprint=[self fingerprint:out];return g; }
@end

