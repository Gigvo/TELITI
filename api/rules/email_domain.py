"""Email-domain rules — MVP_PLAN.md step 1.4.

Owns three slots of the feature vector:

- `email_free_provider`   contact address is a consumer mailbox, not a company domain
- `email_absent`          no email address offered at all
- `email_domain_mismatch` corporate-looking domain that does not match the company named

## Availability

All three report `available=False` when the input carries EMSCAD redaction
placeholders. On that corpus the contact details were stripped before we ever saw
the text, so "no email" would be a fact about the dataset rather than about the ad.
See `api/rules/base.py` for what the fusion training code must do with that.
"""

from __future__ import annotations

from api.ingest import (
    IngestResult,
    Match,
    acronym_of,
    email_domain,
    normalize_for_match,
    registrable_label,
    url_host,
)
from api.rules.base import Rule, RuleOutcome
from api.rules.lexicon import (
    DISPOSABLE_EMAIL_PROVIDERS,
    FREE_EMAIL_PROVIDERS,
    LINK_AGGREGATOR_ALLOWLIST,
    URL_SHORTENERS,
)
from api.schemas import RuleCategory

CATEGORY = RuleCategory.COMPANY

_LABEL_FREE = ("Kontak memakai email gratis, bukan domain perusahaan",
               "Contact uses a free email provider, not a company domain")
_LABEL_DISPOSABLE = ("Kontak memakai email sekali pakai",
                     "Contact uses a disposable email service")
_LABEL_ABSENT = ("Tidak ada alamat email resmi untuk melamar",
                 "No formal email address provided for applying")
_LABEL_MISMATCH = ("Domain email tidak cocok dengan nama perusahaan yang disebut",
                   "Email domain does not match the company name given")

_REDACTED = "Detail kontak telah disensor pada sumber data ini."


class EmailDomainRule(Rule):
    feature_ids = ("email_free_provider", "email_absent", "email_domain_mismatch")

    def evaluate(self, ctx: IngestResult) -> list[RuleOutcome]:
        if ctx.has_redaction_placeholders:
            return [
                self._unavailable(fid, *self._labels(fid), CATEGORY, _REDACTED)
                for fid in self.feature_ids
            ]

        return [
            self._eval_free_provider(ctx),
            self._eval_absent(ctx),
            self._eval_mismatch(ctx),
        ]

    # -- individual features -------------------------------------------------

    def _eval_free_provider(self, ctx: IngestResult) -> RuleOutcome:
        for match in ctx.emails:
            label = registrable_label(email_domain(match.text))

            if label in DISPOSABLE_EMAIL_PROVIDERS:
                return RuleOutcome(
                    feature_id="email_free_provider",
                    severity=0.95,
                    label_id=_LABEL_DISPOSABLE[0],
                    label_en=_LABEL_DISPOSABLE[1],
                    category=CATEGORY,
                    evidence=match.text,
                    span=match.to_span(),
                )

            if label in FREE_EMAIL_PROVIDERS:
                # A free mailbox is a real signal but a weak one on its own: small
                # Indonesian businesses and campus career contacts legitimately use
                # Gmail. The fusion model learns how much it is worth; our job is
                # to report it honestly, not to pre-judge it.
                return RuleOutcome(
                    feature_id="email_free_provider",
                    severity=0.55,
                    label_id=_LABEL_FREE[0],
                    label_en=_LABEL_FREE[1],
                    category=CATEGORY,
                    evidence=match.text,
                    span=match.to_span(),
                )

        return self._clean("email_free_provider", *_LABEL_FREE, CATEGORY)

    def _eval_absent(self, ctx: IngestResult) -> RuleOutcome:
        if ctx.emails:
            return self._clean("email_absent", *_LABEL_ABSENT, CATEGORY)

        # Severity depends on what else is on offer.
        #
        # If the ad points at a genuine company site, the absence of an email is not
        # weak evidence — it is NO evidence. Applying through a career page is how
        # real companies hire. Assigning it a small non-zero severity would put a
        # warning card on a spotless posting, which is precisely the false positive
        # section 3.6 tells us to suppress.
        # Link aggregators COUNT as a real route here, unlike in the shortener rule.
        # A Linktree is a published, attributable page — Indonesian campus career
        # centres and SMEs route real postings through one. An anonymous shortener
        # hides its destination; an aggregator does not. Excluding aggregators here
        # made `email_absent` fire on a legitimate career-centre post.
        has_real_site = any(url_host(u.text) not in URL_SHORTENERS for u in ctx.urls)
        if has_real_site:
            return self._clean("email_absent", *_LABEL_ABSENT, CATEGORY)

        # Only a phone or a shortened link: unusual, but there is still some route.
        # No route at all: an ad that wants applicants but names no way to reach it.
        severity = 0.35 if (ctx.urls or ctx.phones) else 0.55

        return RuleOutcome(
            feature_id="email_absent",
            severity=severity,
            label_id=_LABEL_ABSENT[0],
            label_en=_LABEL_ABSENT[1],
            category=CATEGORY,
            evidence="Tidak ditemukan alamat email pada teks lowongan.",
        )

    def _eval_mismatch(self, ctx: IngestResult) -> RuleOutcome:
        corporate = [
            m for m in ctx.emails
            if registrable_label(email_domain(m.text)) not in FREE_EMAIL_PROVIDERS
            and registrable_label(email_domain(m.text)) not in DISPOSABLE_EMAIL_PROVIDERS
        ]
        if not corporate or not ctx.companies:
            # Nothing to compare. Not a mismatch, and not evidence of anything.
            return self._clean("email_domain_mismatch", *_LABEL_MISMATCH, CATEGORY)

        for match in corporate:
            label = registrable_label(email_domain(match.text))
            if not any(self._matches_company(label, c) for c in ctx.companies):
                return RuleOutcome(
                    feature_id="email_domain_mismatch",
                    severity=0.60,
                    label_id=_LABEL_MISMATCH[0],
                    label_en=_LABEL_MISMATCH[1],
                    category=CATEGORY,
                    evidence=match.text,
                    span=match.to_span(),
                )

        return self._clean("email_domain_mismatch", *_LABEL_MISMATCH, CATEGORY)

    # -- matching ------------------------------------------------------------

    @staticmethod
    def _matches_company(domain_label: str, company: Match) -> bool:
        """Does `domain_label` plausibly belong to `company`?

        Deliberately generous. A false "mismatch" accuses a real company of
        impersonating itself, which is the expensive error here (section 3.6).
        """
        # Drop the legal-entity prefix: "PT Teknologi Nusantara" -> "Teknologi Nusantara".
        bare = company.text.split(None, 1)[1] if " " in company.text else company.text
        normalized = normalize_for_match(bare)
        domain_label = domain_label.lower()

        if not normalized or not domain_label:
            return True

        if domain_label in normalized or normalized in domain_label:
            return True

        # "bca.co.id" for "Bank Central Asia".
        if domain_label == acronym_of(bare):
            return True

        # Any single word of the company name carrying real weight, e.g.
        # "nusantara.co.id" for "PT Teknologi Nusantara".
        for token in bare.split():
            token_norm = normalize_for_match(token)
            if len(token_norm) >= 4 and token_norm in domain_label:
                return True

        return False

    @staticmethod
    def _labels(feature_id: str) -> tuple[str, str]:
        return {
            "email_free_provider": _LABEL_FREE,
            "email_absent": _LABEL_ABSENT,
            "email_domain_mismatch": _LABEL_MISMATCH,
        }[feature_id]
