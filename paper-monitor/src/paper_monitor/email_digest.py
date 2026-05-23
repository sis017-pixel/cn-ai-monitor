
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import logging
import os
import re
import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Any

LOG=logging.getLogger("email_digest")

DATE_RE=re.compile(r"(\d{4}-\d{2}-\d{2})")
WS_RE=re.compile(r"\s+")


def env(name:str,default:str="")->str:
    return (os.getenv(name) or default).strip()


def env_int(name:str,default:int)->int:
    try:
        return int(env(name,str(default)))
    except ValueError:
        return default


def env_bool(name:str,default:bool)->bool:
    value=env(name,"true" if default else "false").lower()
    return value in {"1","true","yes","y","on"}


def clean(value:Any)->str:
    if value is None:
        return ""
    if isinstance(value,list):
        value=" ".join(str(x) for x in value)
    return WS_RE.sub(" ",str(value)).strip()


def short(value:Any,n:int=280)->str:
    text=clean(value)
    return text if len(text)<=n else text[:n-1].rstrip()+"..."


def root()->Path:
    here=Path.cwd().resolve()
    for p in [here]+list(here.parents):
        if (p/"pyproject.toml").exists():
            return p
    return here


def latest(pattern:str)->Path|None:
    d=root()/"data"/"processed"
    files=list(d.glob(pattern)) if d.exists() else []
    if not files:
        return None

    def key(p:Path)->tuple[str,float]:
        m=DATE_RE.search(p.name)
        return (m.group(1) if m else "",p.stat().st_mtime)

    return sorted(files,key=key)[-1]


def date_tag(path:Path)->str:
    m=DATE_RE.search(path.name)
    return m.group(1) if m else dt.date.today().isoformat()


def load_digest(path:Path)->dict[str,Any]:
    with path.open("r",encoding="utf-8") as f:
        data=json.load(f)
    if not isinstance(data,dict):
        raise ValueError("final_digest JSON must be an object")
    return data


def bucket(digest:dict[str,Any],name:str)->list[dict[str,Any]]:
    buckets=digest.get("buckets")
    if not isinstance(buckets,dict):
        return []
    values=buckets.get(name)
    return [x for x in values if isinstance(x,dict)] if isinstance(values,list) else []


def item_title(item:dict[str,Any])->str:
    return clean(item.get("title")) or "(untitled)"


def item_url(item:dict[str,Any])->str:
    return clean(item.get("url"))


def item_summary(item:dict[str,Any])->str:
    s=item.get("summary")
    if isinstance(s,dict):
        bullets=s.get("summary_bullets")
        if isinstance(bullets,list) and bullets:
            return short(bullets[0],320)
        return short(s.get("why_it_matters") or s.get("technical_contribution"),320)
    return ""


def evidence(item:dict[str,Any])->str:
    verdict=clean(item.get("claude_verdict"))
    if verdict:
        return "Claude verdict: "+verdict

    inst=item.get("institutions") or []
    countries=item.get("countries") or []
    pieces=[]

    if isinstance(inst,list) and inst:
        pieces.append("institutions: "+", ".join(str(x) for x in inst[:3]))
    if isinstance(countries,list) and countries:
        pieces.append("countries: "+", ".join(str(x) for x in countries[:3]))

    return "; ".join(pieces) or clean(item.get("bucket_reason")) or "metadata only"


def scores(item:dict[str,Any])->str:
    return "significance="+str(item.get("significance_score"))+", AI="+str(item.get("ai_relevance_score"))


def run_url()->str:
    if env("FULL_DIGEST_URL"):
        return env("FULL_DIGEST_URL")

    server=os.getenv("GITHUB_SERVER_URL")
    repo=os.getenv("GITHUB_REPOSITORY")
    run_id=os.getenv("GITHUB_RUN_ID")

    if server and repo and run_id:
        return server.rstrip("/")+"/"+repo+"/actions/runs/"+run_id

    return ""


def subject(digest:dict[str,Any],tag:str)->str:
    c=digest.get("counts") if isinstance(digest.get("counts"),dict) else {}
    return f'{env("DIGEST_EMAIL_SUBJECT_PREFIX","CN AI Monitor")} - {tag}: {c.get("verified_cn",0)} verified, {c.get("ecosystem_signal",0)} ecosystem, {c.get("insufficient_evidence",0)} audit'


def hlink(title:str,url:str)->str:
    if url:
        return f'<a href="{html.escape(url,quote=True)}">{html.escape(title)}</a>'
    return html.escape(title)


def html_section(title:str,items:list[dict[str,Any]],limit:int)->str:
    out=[f'<h2 style="font-size:18px;margin:24px 0 10px;">{html.escape(title)}</h2>']

    if not items:
        out.append('<p style="color:#666;">No papers in this section.</p>')
        return "\n".join(out)

    for item in items[:limit]:
        out.append('<div style="border-left:4px solid #d0d7de;padding:10px 0 10px 14px;margin:12px 0;">')
        out.append('<div style="font-weight:700;font-size:15px;">'+hlink(item_title(item),item_url(item))+'</div>')
        out.append('<div style="font-size:13px;color:#57606a;margin-top:4px;">'+html.escape(scores(item))+'</div>')
        out.append('<div style="font-size:13px;color:#57606a;margin-top:4px;">'+html.escape(evidence(item))+'</div>')

        summ=item_summary(item)
        if summ:
            out.append('<p style="font-size:14px;line-height:1.45;margin:8px 0 0;">'+html.escape(summ)+'</p>')

        out.append('</div>')

    if len(items)>limit:
        out.append(f'<p style="color:#57606a;font-size:13px;">+{len(items)-limit} more in the full digest.</p>')

    return "\n".join(out)


def build_html(digest:dict[str,Any],tag:str,url:str,limit:int)->str:
    c=digest.get("counts") if isinstance(digest.get("counts"),dict) else {}
    verified=bucket(digest,"verified_cn")
    ecosystem=bucket(digest,"ecosystem_signal")
    insufficient=bucket(digest,"insufficient_evidence")

    button=""
    if url:
        button=f'<p style="margin:24px 0;"><a href="{html.escape(url,quote=True)}" style="background:#0969da;color:#fff;text-decoration:none;padding:12px 18px;border-radius:6px;font-weight:700;display:inline-block;">View full digest</a></p>'

    method=clean(digest.get("method_note")) or "Verified China-affiliated papers require institution/country/affiliation evidence. Chinese model usage alone is treated as an ecosystem signal."

    return f'''<!doctype html>
<html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#f6f8fa;margin:0;padding:24px;">
<div style="max-width:820px;margin:0 auto;background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:28px;">
<h1 style="font-size:24px;margin:0 0 8px;">CN AI Monitor Digest - {html.escape(tag)}</h1>
<p style="color:#57606a;margin:0 0 22px;">Automated evidence-aware digest.</p>
<div style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:16px;margin:14px 0 18px;">
<p><strong>Verified China-affiliated AI papers:</strong> {html.escape(str(c.get("verified_cn",len(verified))))}</p>
<p><strong>China ecosystem signals:</strong> {html.escape(str(c.get("ecosystem_signal",len(ecosystem))))}</p>
<p><strong>Insufficient evidence / audit appendix:</strong> {html.escape(str(c.get("insufficient_evidence",len(insufficient))))}</p>
</div>
<p style="font-size:14px;line-height:1.5;"><strong>Method note:</strong> {html.escape(method)}</p>
{button}
{html_section("Verified China-Affiliated AI Papers",verified,limit)}
{html_section("China Ecosystem Signals",ecosystem,limit)}
{html_section("Insufficient Evidence / Audit Appendix",insufficient,min(3,limit))}
<hr style="border:none;border-top:1px solid #d0d7de;margin:28px 0;">
<p style="font-size:12px;color:#57606a;">Generated automatically by cn-ai-monitor, an project by SIYU SHEN.</p>
</div></body></html>'''


def text_section(title:str,items:list[dict[str,Any]],limit:int)->str:
    lines=["",title,"-"*len(title)]

    if not items:
        lines.append("No papers in this section.")
        return "\n".join(lines)

    for i,item in enumerate(items[:limit],1):
        lines.append(f"{i}. {item_title(item)}")
        if item_url(item):
            lines.append("   URL: "+item_url(item))
        lines.append("   Scores: "+scores(item))
        lines.append("   Evidence: "+evidence(item))
        if item_summary(item):
            lines.append("   Summary: "+item_summary(item))

    if len(items)>limit:
        lines.append(f"+{len(items)-limit} more in the full digest.")

    return "\n".join(lines)


def build_text(digest:dict[str,Any],tag:str,url:str,limit:int)->str:
    c=digest.get("counts") if isinstance(digest.get("counts"),dict) else {}
    verified=bucket(digest,"verified_cn")
    ecosystem=bucket(digest,"ecosystem_signal")
    insufficient=bucket(digest,"insufficient_evidence")

    lines=[
        "CN AI Monitor Digest - "+tag,
        "",
        "Verified China-affiliated AI papers: "+str(c.get("verified_cn",len(verified))),
        "China ecosystem signals: "+str(c.get("ecosystem_signal",len(ecosystem))),
        "Insufficient evidence / audit appendix: "+str(c.get("insufficient_evidence",len(insufficient))),
        "",
        "Method note: "+clean(digest.get("method_note")),
    ]

    if url:
        lines+=["","Full digest: "+url]

    lines.append(text_section("Verified China-Affiliated AI Papers",verified,limit))
    lines.append(text_section("China Ecosystem Signals",ecosystem,limit))
    lines.append(text_section("Insufficient Evidence / Audit Appendix",insufficient,min(3,limit)))
    return "\n".join(lines)


def attach_md(msg:EmailMessage,path:Path|None)->None:
    if not path or not path.exists():
        return
    msg.add_attachment(path.read_bytes(),maintype="text",subtype="markdown",filename=path.name)


def send(msg:EmailMessage,host:str,port:int,user:str,password:str)->None:
    context=ssl.create_default_context()
    with smtplib.SMTP_SSL(host,port,context=context) as server:
        server.login(user,password)
        server.send_message(msg)


def parse_args(argv:list[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser(description="Send concise final digest email through Gmail SMTP.")
    p.add_argument("--digest-json",default=None)
    p.add_argument("--digest-md",default=None)
    p.add_argument("--to",default=None)
    p.add_argument("--full-url",default=None)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--log-level",default="INFO")
    return p.parse_args(argv)


def main(argv:list[str]|None=None)->int:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower()!="utf-8":
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    args=parse_args(argv)
    logging.basicConfig(level=getattr(logging,args.log_level.upper(),logging.INFO),format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.digest_json:
        digest_path=Path(args.digest_json).resolve()
    else:
        found=latest("final_digest_*.json")
        if not found:
            LOG.error("No final_digest_*.json found. Run publish.py first.")
            return 2
        digest_path=found.resolve()

    digest=load_digest(digest_path)
    tag=date_tag(digest_path)

    md_path=Path(args.digest_md).resolve() if args.digest_md else digest_path.with_suffix(".md")
    if not md_path.exists():
        md_path=latest("final_digest_*.md")

    user=env("SMTP_USER")
    password=env("SMTP_APP_PASSWORD")
    recipient=args.to or env("DIGEST_EMAIL_TO")

    if not user or not password or not recipient:
        LOG.error("Missing SMTP_USER, SMTP_APP_PASSWORD, or DIGEST_EMAIL_TO.")
        return 2

    full_url=args.full_url or run_url()
    top_n=env_int("EMAIL_DIGEST_TOP_N",8)

    msg=EmailMessage()
    msg["Subject"]=subject(digest,tag)
    msg["From"]=formataddr((env("DIGEST_EMAIL_FROM_NAME","CN AI Digest"),user))
    msg["To"]=recipient
    msg.set_content(build_text(digest,tag,full_url,top_n))
    msg.add_alternative(build_html(digest,tag,full_url,top_n),subtype="html")

    if env_bool("ATTACH_FULL_DIGEST",True):
        attach_md(msg,md_path)

    LOG.info("Recipient: %s",recipient)
    LOG.info("Sender: %s",user)
    LOG.info("Full digest URL: %s",full_url or "not set; attachment/run artifact will be used")
    LOG.info("Attachment: %s",md_path if md_path and md_path.exists() else "none")
    LOG.info("Subject: %s",msg["Subject"])

    if args.dry_run:
        # After adding an attachment, EmailMessage becomes multipart/mixed.
        # Calling msg.get_content() on multipart messages raises KeyError.
        # Print the plain-text preview directly instead.
        plain_part=msg.get_body(preferencelist=("plain",))
        if plain_part is not None:
            print(plain_part.get_content()[:2500])
        else:
            print(msg.as_string()[:2500])
        LOG.info("Dry run enabled; not sending email.")
        return 0

    send(msg,env("SMTP_HOST","smtp.gmail.com"),env_int("SMTP_PORT",465),user,password)
    LOG.info("Email sent successfully to %s",recipient)
    return 0


if __name__=="__main__":
    sys.exit(main())
