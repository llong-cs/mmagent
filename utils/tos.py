import bytedtos
import hashlib
import random
import string

# os.environ['CONSUL_HTTP_HOST'] = "10.54.129.29"
# os.environ['CONSUL_HTTP_PORT'] = 2280
# PSM、Cluster、Idc、Accesskey 和 Bucket 可在 TOS 用户平台 > Bucket 详情 > 概览页中查找。具体查询方式详见方式二：通过 “psm+idc” 访问 TOS 桶 。

server = "cn"
if server == "cn":
    ak = "YFPD6L54IEAAU421YMSG"
    bucket_name = "vlm-agent"
    tos_psm = "toutiao.tos.tosapi"
    tos_cluster = "default"
    tos_idc = "hl"
    base_url = "https://tosv.byted.org/obj/vlm-agent/"
elif server == "va":
    ak = "BX2M82TQJ7UVTYXYO19Z"
    bucket_name = "vlm-agent-benchmarking-us"
    tos_psm = "toutiao.tos.tosapi"
    tos_cluster = "default"
    tos_idc = "maliva"
    base_url = "https://tosv-va.tiktok-row.org/obj/vlm-agent-benchmarking-us/"
else:
    raise ValueError(f"Invalid server: {server}")

tos_client = bytedtos.Client(
    bucket_name, ak, service=tos_psm, cluster=tos_cluster, idc=tos_idc
)


def get_hash_key(text):
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode("utf-8"))
    hash_int = int.from_bytes(md5_hash.digest(), byteorder="big")
    return abs(hash_int) % (10**8)


def generate_random_clip_name(length=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def upload_one_sample(file, do_upload=True):
    try:
        obj_key = generate_random_clip_name()
        obj_url = base_url + obj_key
        if do_upload:
            content = open(file, "rb")
            resp = tos_client.put_object(obj_key, content)
            resp_code = int(resp.status_code)
            if resp_code != 200:
                print(f"Upoload error code: {resp_code}")
                return -1, ""
    except bytedtos.TosException as e:
        print(
            "Upload failed. code: {}, request_id: {}, message: {}".format(
                e.code, e.request_id, e.msg
            )
        )
        return -1, ""
    except Exception as e:
        print(f"Other error: {e}")
        return -1, ""
    return obj_url

if __name__ == "__main__":
    upload_one_sample("/Users/bytedance/Downloads/tmp6nvagz76.wav")
