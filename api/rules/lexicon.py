"""Static vocabularies for the rule layer.

Kept in Python rather than YAML because these are stable, developer-maintained
reference lists. The Indonesian *risk phrase* lexicon is different — it is produced
by the annotators as a by-product of building the evaluation set, so it lives in
`data/reference/risk_phrases_id.yaml` and lands with step 2.4.
"""

from __future__ import annotations

#: Consumer mailbox providers. A recruiter at a real company writes from a company
#: domain; a free provider is not proof of fraud but is a genuine signal, and it is
#: the example the concept paper itself gives (section 2.3).
FREE_EMAIL_PROVIDERS = frozenset(
    {
        "gmail", "googlemail", "yahoo", "ymail", "rocketmail", "hotmail", "outlook",
        "live", "msn", "aol", "icloud", "me", "mac", "gmx", "yandex", "mail",
        "zoho", "protonmail", "proton", "tutanota", "inbox", "hushmail", "fastmail",
    }
)

#: Throwaway mailbox services. Materially worse than a personal Gmail: there is no
#: legitimate reason for an employer to recruit from an address that self-destructs.
DISPOSABLE_EMAIL_PROVIDERS = frozenset(
    {
        "mailinator", "guerrillamail", "10minutemail", "tempmail", "temp-mail",
        "throwawaymail", "yopmail", "trashmail", "sharklasers", "getnada",
        "maildrop", "dispostable", "fakeinbox", "mohmal",
    }
)

#: Link shorteners. They hide the destination, which is the entire reason a scam
#: uses one. `s.id` is Indonesia's most common shortener and appears constantly in
#: local scam posts.
URL_SHORTENERS = frozenset(
    {
        "bit.ly", "bitly.com", "tinyurl.com", "s.id", "cutt.ly", "ow.ly", "t.co",
        "goo.gl", "rebrand.ly", "is.gd", "buff.ly", "shorturl.at", "rb.gy",
        "gg.gg", "bit.do", "adf.ly", "shorte.st", "tiny.cc", "v.gd", "urlz.fr",
    }
)

#: Deliberately NOT treated as shorteners. Indonesian SMEs and legitimate campus
#: career centres use these constantly for real postings; flagging them would
#: generate exactly the false positives section 3.6 tells us to avoid.
LINK_AGGREGATOR_ALLOWLIST = frozenset({"linktr.ee", "linktree.com", "lynk.id", "bio.link"})

#: Hosts that indicate a chat handoff rather than a formal application route.
WHATSAPP_HOSTS = frozenset({"wa.me", "api.whatsapp.com", "chat.whatsapp.com", "whatsapp.com"})
TELEGRAM_HOSTS = frozenset({"t.me", "telegram.me", "telegram.dog", "telegram.org"})

#: Indonesian phrasing that routes an applicant into a private chat.
#: WhatsApp phrasing is listed separately from Telegram phrasing on purpose — see
#: `api/rules/contact_channel.py` for why the two carry very different weights.
WHATSAPP_PHRASES = (
    "hubungi via whatsapp", "hubungi via wa", "chat wa", "wa ke", "whatsapp ke",
    "hubungi wa", "langsung wa", "japri wa", "chat langsung ke wa",
)

TELEGRAM_PHRASES = (
    "via telegram", "interview via telegram", "hubungi telegram", "daftar via telegram",
    "gabung telegram", "chat telegram", "wawancara via telegram", "telegram admin",
)

#: Phrases indicating the interview or hiring decision happens entirely in chat.
#: A real employer does not conclude a hire inside a messaging thread.
CHAT_HIRING_PHRASES = (
    "interview via chat", "wawancara via chat", "interview online via",
    "langsung kerja tanpa interview", "tanpa interview", "diterima tanpa wawancara",
)

#: Generic "contact the admin" phrasing — a real posting names a company or a role,
#: not an anonymous "admin".
ANONYMOUS_RECRUITER_PHRASES = ("hubungi admin", "chat admin", "hubungi kak", "hubungi bapak/ibu")
