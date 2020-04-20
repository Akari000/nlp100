# 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．

import json
import re
text = ""


## TODO メディアファイルは [File: ]という形のもの
## 英字だけ抜き出す


pattern = r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+"
with open('../data/jawiki-England.json', "r") as f:
    data = json.loads(f.read())
    text = data["text"]

url_list = re.findall(pattern, text)

for url in url_list:
    print(url)

# http://esa.un.org/unpd/wpp/Excel-Data/population.htm
# http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=
# http://www.mod.go.jp/msdf/formal/info/news/200808/082001.html
# http://www.cnn.co.jp/world/35023094.html
# http://yoshio-kusano.sakura.ne.jp/nakayamakouen6newpage3.html
# http://www.globalpowereurope.eu/
# http://www.raf.mod.uk/legalservices/p3chp29.htm
# http://www.mod.uk/DefenceInternet/AboutDefence/Organisation/KeyFactsAboutDefence/DefenceSpending.htm
# http://www.mod.uk/NR/rdonlyres/6FBA7459-7407-4B85-AA47-7063F1F22461/0/modara_0405_s1_resources.pdf
# http://www.asahi.com/international/update/1201/TKY201111300900.html?ref=reca
# http://mainichi.jp/select/world/europe/news/20111130k0000e030066000c.html
# http://www.imf.org/external/pubs/ft/weo/2014/01/weodata/weorept.aspx?pr.x=71&pr.y=15&sy=2013&ey=2019&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=NGDPD%2CNGDPDPC&grp=0&a=
# http://www.atkearney.com/documents/10192/4461492/Global+Cities+Present+and+Future-GCI+2014.pdf/3628fd7d-70be-41bf-99d6-4c8eaf984cd5
# http://www.sh.xinhuanet.com/shstatics/images2013/IFCD2013_En.pdf
# https://www.bis.org/publ/rpfx13fx.pdf
# http://www.bcg.com/expertise_impact/publications/PublicationDetails.aspx?id=tcm:12-107081
# http://sankei.jp.msn.com/economy/business/080830/biz0808301850007-n1.htm
# https://www.gov.uk/government/news/introduction-of-same-sex-marriage-at-british-consulates-overseas.ja
# http://www.royal.gov.uk/
# http://www.direct.gov.uk/
# https://www.gov.uk/government/organisations/prime-ministers-office-10-downing-street
# https://www.gov.uk/government/world/japan.ja
# https://www.gov.uk/government/world/organisations/british-embassy-tokyo.ja
# https://www.gov.uk/government/organisations/uk-visas-and-immigration
# http://www.vfsglobal.co.uk/japan/Japanese/
# http://www.mofa.go.jp/mofaj/area/uk/
# http://www.uk.emb-japan.go.jp/jp/index.html
# http://www.visitbritain.com/ja/JP/
# http://www.jetro.go.jp/world/europe/uk/