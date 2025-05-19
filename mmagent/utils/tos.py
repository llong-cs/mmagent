import bytedtos
import hashlib
import random
import string
import os
import json

# os.environ['CONSUL_HTTP_HOST'] = "10.54.129.29"
# os.environ['CONSUL_HTTP_PORT'] = 2280
# PSM、Cluster、Idc、Accesskey 和 Bucket 可在 TOS 用户平台 > Bucket 详情 > 概览页中查找。具体查询方式详见方式二：通过 “psm+idc” 访问 TOS 桶 。

try:
    tos_config = json.load(open("/mnt/bn/videonasi18n/longlin.kylin/mmagent/tos_config.json"))

    server = "va"
    if server not in ["cn", "va"]:
        raise ValueError(f"Invalid server: {server}")

    tos_config = tos_config[server]
    ak = tos_config["ak"]
    bucket_name = tos_config["bucket_name"]
    tos_psm = tos_config["tos_psm"]
    tos_cluster = tos_config["tos_cluster"]
    tos_idc = tos_config["tos_idc"]
    base_url = tos_config["base_url"]

    tos_client = bytedtos.Client(
        bucket_name, ak, service=tos_psm, cluster=tos_cluster, idc=tos_idc
    )
except:
    pass


def get_hash_key(text):
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode("utf-8"))
    hash_int = int.from_bytes(md5_hash.digest(), byteorder="big")
    return abs(hash_int) % (10**8)


def generate_random_clip_name(length=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def upload_one_sample(file, obj_key=None, do_upload=True):
    try:
        if not obj_key:
            obj_key = generate_random_clip_name()
        obj_url = base_url + obj_key
        if do_upload:
            content = open(file, "rb")
            resp = tos_client.put_object(obj_key, content)
            resp_code = int(resp.status_code)
            if resp_code != 200:
                print(f"Upoload error code: {resp_code}")
                return "", ""
    except bytedtos.TosException as e:
        print(
            "Upload failed. code: {}, request_id: {}, message: {}".format(
                e.code, e.request_id, e.msg
            )
        )
        return "", ""
    except Exception as e:
        print(f"Other error: {e}")
        return "", ""
    return obj_url, obj_key
    
def download_one_sample(file, obj_key):
    try:
        resp = tos_client.get_object(obj_key)
        content = resp.data
        with open(file, "wb") as f:
            f.write(content)
            # Flush to ensure all data is written to disk
            f.flush()
            # Force the operating system to write the data to disk
            os.fsync(f.fileno())
        return file
    except bytedtos.TosException as e:
        print(
            "Download failed. code: {}, request_id: {}, message: {}".format(
                e.code, e.request_id, e.msg
            )
        )
        return None
    except Exception as e:
        print(f"Other error: {e}")
        return None
    
def list_all_objects(prefix_of_obj, target_delimiter="/", target_start_after="", target_max_keys=1000):
    try:
        resp = tos_client.list_prefix(prefix_of_obj, target_delimiter, target_start_after, target_max_keys)
        data = resp.json["payload"]
        commonPrefix = data["commonPrefix"]
        objects = data["objects"]
        file_list = [item.get("key").split("/")[-1].split(".")[0] for item in objects]
        return file_list
    except bytedtos.TosException as e:
        print(f"List objects failed. code: {e.code}, request_id: {e.request_id}, message: {e.msg}")
        return []
    except Exception as e:
        print(f"Other error: {e}")
        return []

if __name__ == "__main__":
    upload_one_sample("/Users/bytedance/Downloads/tmp6nvagz76.wav")
