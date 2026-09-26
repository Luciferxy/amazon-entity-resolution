import re
from anyascii import anyascii

LEGAL = {"inc":"inc","incorporated":"inc","corp":"corp","corporation":"corp","co":"co","company":"co",
         "llc":"llc","llp":"llp","ltd":"ltd","limited":"ltd","pvt":"pvt","private":"pvt","plc":"plc",
         "sarl":"sarl","sas":"sas","sa":"sa","eurl":"eurl","sci":"sci"}
LEGAL_SET = set(LEGAL.values())
ADDR = {"road":"rd","street":"st","avenue":"ave","av":"ave","drive":"dr","place":"pl","terrace":"ter",
        "boulevard":"blvd","bd":"blvd","lane":"ln","court":"ct","highway":"hwy","township":"twp",
        "apartment":"apt","suite":"ste","building":"bldg","floor":"fl","sector":"sec",
        "near":"nr","opposite":"opp","north":"n","south":"s","east":"e","west":"w"}
STATE = {
 "us": {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
        "connecticut":"ct","delaware":"de","district of columbia":"dc","florida":"fl","georgia":"ga",
        "hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks",
        "kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma",
        "michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt",
        "nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm",
        "new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
        "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc",
        "south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt",
        "virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy"},
 "india": {"andhra pradesh":"ap","arunachal pradesh":"ar","assam":"as","bihar":"br",
        "chhattisgarh":"cg","chattisgarh":"cg","goa":"ga","gujarat":"gj","haryana":"hr",
        "himachal pradesh":"hp","jharkhand":"jh","karnataka":"ka","kerala":"kl",
        "madhya pradesh":"mp","maharashtra":"mh","manipur":"mn","meghalaya":"ml","mizoram":"mz",
        "nagaland":"nl","odisha":"od","orissa":"od","punjab":"pb","rajasthan":"rj","sikkim":"sk",
        "tamil nadu":"tn","tamilnadu":"tn","telangana":"ts","tripura":"tr","uttar pradesh":"up",
        "uttarakhand":"uk","uttaranchal":"uk","west bengal":"wb",
        "andaman and nicobar islands":"an","chandigarh":"ch",
        "dadra and nagar haveli and daman and diu":"dd","delhi":"dl","jammu and kashmir":"jk",
        "ladakh":"la","lakshadweep":"ld","puducherry":"py","pondicherry":"py"}}
STATE_RE = {c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b")
            for c, d in STATE.items()}
NULLS = {"null", "none", "nan", "na"}
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|biz|in|co\.in|fr)$")

def base(s):
    if not isinstance(s, str): return ""
    s = anyascii(s).lower().strip().replace("&", " and ")
    s = DOMAIN_RE.sub(r"\1", s)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)        # 45ND -> 45
    return re.sub(r"[^a-z0-9 ]+", " ", s)

def norm_name(s):
    t = [LEGAL.get(w, w) for w in base(s).split()]
    core = [w for w in t if w not in LEGAL_SET] or t
    return " ".join(t), " ".join(core), "".join(core)    # full, core, nospace

def norm_addr(s, country):
    s = base(s)
    if country in STATE_RE:                               # other countries: generic path
        s = STATE_RE[country].sub(lambda m: STATE[country][m.group(1)], s)
    t = [ADDR.get(w, w) for w in s.split() if w not in NULLS]
    nums = {w for w in t if w.isdigit()}
    return " ".join(sorted(t)), nums, {w for w in nums if len(w) in (5, 6)}
