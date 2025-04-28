include "base.thrift"

namespace py face_processing
namespace go face_processing
namespace cpp face_processing

struct Face {
    1: i64 frame_id,
    2: list<i64> bounding_box,
    3: list<double> face_emb,
    4: i64 cluster_id,
    5: optional map<string, string> extra_data //正脸/侧脸可以作为extra_data中的一个字段。
}

struct SingleGetFaceRequest {
    1: list<string> frames,
    254: optional map<string, string> extra_data,
    255: base.Base Base
}

struct SingleGetFaceResponse {
    1: list<Face> faces,
    254: optional map<string, string> extra_data,
    255: base.BaseResp BaseResp
}

struct SingleClusterFaceRequest {
    1: list<Face> faces,
    254: optional map<string, string> extra_data,
    255: base.Base Base
}

struct SingleClusterFaceResponse {
    1: list<Face> faces,
    254: optional map<string, string> extra_data,
    255: base.BaseResp BaseResp
}

service FaceService {
    SingleGetFaceResponse SingleGetFace(1: SingleGetFaceRequest request)
    SingleGetFaceResponse SingleClusterFace(1: SingleClusterFaceRequest request)
}