"""Contact-channel rules — MVP_PLAN.md step 1.4.

Owns two slots of the feature vector:

- `contact_messaging_only`  the only way to apply is a private chat handoff
- `url_shortener`           the destination link is hidden behind a shortener

## The calibration that matters most here

A naive version of this rule flags WhatsApp and produces a false positive on a large
share of legitimate Indonesian job ads. In Indonesia WhatsApp *is* ordinary business
communication — SMEs, campus career centres and real recruiters all use it, and an
HRD WhatsApp number in a posting is unremarkable.

Telegram is different. It is not a normal Indonesian business channel, and
"interview via Telegram" is one of the most reliable markers in the scam reporting
the concept paper cites (section 1.1). So the two are weighted very differently:
WhatsApp-only is mild, Telegram is strong.

The rule also only fires when chat is the *only* route. An ad offering a company
email or a career page alongside a WhatsApp number is behaving normally.
"""

from __future__ import annotations

from api.ingest import IngestResult, Match, email_domain, registrable_label, url_host
from api.rules.base import Rule, RuleOutcome
from api.rules.lexicon import (
    ANONYMOUS_RECRUITER_PHRASES,
    CHAT_HIRING_PHRASES,
    DISPOSABLE_EMAIL_PROVIDERS,
    FREE_EMAIL_PROVIDERS,
    LINK_AGGREGATOR_ALLOWLIST,
    TELEGRAM_HOSTS,
    TELEGRAM_PHRASES,
    URL_SHORTENERS,
    WHATSAPP_HOSTS,
    WHATSAPP_PHRASES,
)
from api.schemas import RuleCategory, Span

CATEGORY = RuleCategory.CONTACT

_LABEL_MESSAGING = ("Satu-satunya cara melamar adalah melalui chat pribadi",
                    "The only way to apply is through a private chat")
_LABEL_TELEGRAM = ("Proses rekrutmen diarahkan ke Telegram",
                   "Recruitment process is routed through Telegram")
_LABEL_SHORTENER = ("Tautan disembunyikan di balik pemendek URL",
                    "Destination link is hidden behind a URL shortener")

_REDACTED = "Detail kontak telah disensor pada sumber data ini."

# Severity floors. WhatsApp is ordinary here; Telegram is not.
_SEVERITY_WHATSAPP_ONLY = 0.35
_SEVERITY_TELEGRAM = 0.75
_BOOST_CHAT_HIRING = 0.15
_BOOST_ANONYMOUS_RECRUITER = 0.10


class ContactChannelRule(Rule):
    feature_ids = ("contact_messaging_only", "url_shortener")

    def evaluate(self, ctx: IngestResult) -> list[RuleOutcome]:
        if ctx.has_redaction_placeholders:
            return [
                self._unavailable("contact_messaging_only", *_LABEL_MESSAGING, CATEGORY, _REDACTED),
                self._unavailable("url_shortener", *_LABEL_SHORTENER, CATEGORY, _REDACTED),
            ]
        return [self._eval_messaging_only(ctx), self._eval_shortener(ctx)]

    # -- contact_messaging_only ---------------------------------------------

    def _eval_messaging_only(self, ctx: IngestResult) -> RuleOutcome:
        telegram = self._find_channel(ctx, TELEGRAM_HOSTS, TELEGRAM_PHRASES)
        whatsapp = self._find_channel(ctx, WHATSAPP_HOSTS, WHATSAPP_PHRASES)

        if telegram is None and whatsapp is None:
            return self._clean("contact_messaging_only", *_LABEL_MESSAGING, CATEGORY)

        # A formal route alongside chat means chat is a convenience, not a trap.
        if self._has_formal_application_route(ctx):
            return self._clean("contact_messaging_only", *_LABEL_MESSAGING, CATEGORY)

        if telegram is not None:
            severity = _SEVERITY_TELEGRAM
            evidence_match = telegram
            label_id, label_en = _LABEL_TELEGRAM
        else:
            severity = _SEVERITY_WHATSAPP_ONLY
            evidence_match = whatsapp
            label_id, label_en = _LABEL_MESSAGING

        if any(phrase in ctx.lowered for phrase in CHAT_HIRING_PHRASES):
            severity += _BOOST_CHAT_HIRING
        if any(phrase in ctx.lowered for phrase in ANONYMOUS_RECRUITER_PHRASES):
            severity += _BOOST_ANONYMOUS_RECRUITER

        return RuleOutcome(
            feature_id="contact_messaging_only",
            severity=min(severity, 1.0),
            label_id=label_id,
            label_en=label_en,
            category=CATEGORY,
            evidence=evidence_match.text,
            span=evidence_match.to_span(),
        )

    # -- url_shortener -------------------------------------------------------

    def _eval_shortener(self, ctx: IngestResult) -> RuleOutcome:
        shortened = [m for m in ctx.urls if url_host(m.text) in URL_SHORTENERS]
        if not shortened:
            return self._clean("url_shortener", *_LABEL_SHORTENER, CATEGORY)

        # Worse when it is the only link on offer: nothing else identifies where
        # the applicant is actually being sent.
        others = [
            m for m in ctx.urls
            if url_host(m.text) not in URL_SHORTENERS
            and url_host(m.text) not in LINK_AGGREGATOR_ALLOWLIST
        ]
        severity = 0.45 if others else 0.65

        return RuleOutcome(
            feature_id="url_shortener",
            severity=severity,
            label_id=_LABEL_SHORTENER[0],
            label_en=_LABEL_SHORTENER[1],
            category=CATEGORY,
            evidence=shortened[0].text,
            span=shortened[0].to_span(),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _find_channel(
        ctx: IngestResult, hosts: frozenset[str], phrases: tuple[str, ...]
    ) -> Match | None:
        """Locate a chat route, preferring a real link over a phrase mention."""
        for match in ctx.urls:
            if url_host(match.text) in hosts:
                return match
        for phrase in phrases:
            index = ctx.lowered.find(phrase)
            if index != -1:
                return Match(
                    text=ctx.raw_text[index : index + len(phrase)],
                    start=index,
                    end=index + len(phrase),
                )
        return None

    @staticmethod
    def _has_formal_application_route(ctx: IngestResult) -> bool:
        """A company email or a genuine company website counts as formal.

        A free-provider email does not: "kirim CV ke hrd.rekrutmen@gmail.com" is not
        a formal application route, and treating it as one would let the most common
        Indonesian scam pattern suppress this rule entirely.
        """
        for match in ctx.emails:
            label = registrable_label(email_domain(match.text))
            if label not in FREE_EMAIL_PROVIDERS and label not in DISPOSABLE_EMAIL_PROVIDERS:
                return True

        for match in ctx.urls:
            host = url_host(match.text)
            if (
                host not in URL_SHORTENERS
                and host not in LINK_AGGREGATOR_ALLOWLIST
                and host not in WHATSAPP_HOSTS
                and host not in TELEGRAM_HOSTS
            ):
                return True

        return False
