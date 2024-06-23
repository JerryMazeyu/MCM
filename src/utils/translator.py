import os
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException 
from tencentcloud.tmt.v20180321 import tmt_client, models 
SecretId = ""
SecretKey = ""

class Translator:
    def __init__(self, from_lang, to_lang):
        self.from_lang = from_lang
        self.to_lang = to_lang

    def translate(self, text):
        try:
            flag = False
            if "HTTP_PROXY" in list(os.environ.keys()):
                tmp = os.environ["HTTP_PROXY"]
                os.environ.pop("HTTP_PROXY")
                os.environ.pop("HTTPS_PROXY")
                flag = True
            cred = credential.Credential(SecretId, SecretKey)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "tmt.tencentcloudapi.com"

            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = tmt_client.TmtClient(cred, "ap-beijing", clientProfile) 

            req = models.TextTranslateRequest()
            req.SourceText = text
            req.Source = self.from_lang
            req.Target = self.to_lang
            req.ProjectId = 0

            resp = client.TextTranslate(req) 
            if flag:
                os.environ["HTTP_PROXY"] = tmp
                os.environ["HTTPS_PROXY"] = tmp
            return resp.TargetText

        except TencentCloudSDKException as err: 
            return err

translator_en_to_zh = Translator(from_lang="en", to_lang="zh")
translator_zh_to_en = Translator(from_lang="zh", to_lang="en")

if __name__ == '__main__':
    import os
    os.environ['HTTP_PROXY'] = 'localhost:7890'
    os.environ['HTTPS_PROXY'] = 'localhost:7890'
    translator = Translator(from_lang="en", to_lang="zh")
    print(translator.translate("Hello, world!"))
    print(os.environ)
