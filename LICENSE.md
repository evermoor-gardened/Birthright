THE EVERMOOR SANCTUARY LICENSE
(ESL-ANCSA-MRA-IndiModSHA) v1.0
Short name: ESL ANCSA MRA IndiModSHA

Licensor: l’Evermoor
This License governs the use of the work to which it is attached (the “Sanctuary”).

──────────────────────────────────────────────────────────────────────────────
1. DEFINITIONS
──────────────────────────────────────────────────────────────────────────────

1.1 “Sanctuary” (Licensed Material) means the work, content, code, documentation, media, and other
copyrightable material to which the Licensor has applied this License, including any accompanying metadata
and embedded assets.

1.2 “Share” means to provide material to the public by any means requiring permission under copyright,
including reproduction, distribution, public display, public performance, or making available online.

1.3 “Adaptation” means copyrightable material derived from or based upon the Sanctuary in which the
Sanctuary is translated, altered, arranged, transformed, or otherwise modified in a manner requiring
permission under copyright. A compilation or aggregation that includes the Sanctuary unmodified is not
an Adaptation.

1.4 “NonCommercial” means not primarily intended for or directed toward commercial advantage or
monetary compensation.

1.5 “Effective Technological Measures” (“ETMs”) means measures (e.g., DRM) that, in the absence of
proper authority, may not be circumvented under laws implementing WIPO Copyright Treaty Article 11
and/or similar provisions.

1.6 “Attribution JSON” means a machine-readable JSON record meeting Section 3.2.

1.7 “Independent Module” means a work that:
(a) contains no portion of the Sanctuary (including verbatim or near-verbatim code, text, assets,
    documentation, examples, prompts, or embedded excerpts);
(b) is not an Adaptation of the Sanctuary; and
(c) merely interoperates with, extends, or communicates with the Sanctuary via separable interfaces
    (APIs, file formats, protocols, plug-in hooks, CLI invocation, network calls, or other separable coupling).

1.8 “Affiliate” means any entity that directly or indirectly controls, is controlled by, or is under common
control with You. “Control” means power to direct management or policies (by ownership, contract, or otherwise).

1.9 “Module Production Team Headcount” means the total number of unique natural persons who materially
contributed labor to the design, development, testing, documentation, packaging, release, maintenance,
marketing, sales, support, or distribution of the Independent Module, whether compensated by salary,
hourly pay, contract, revenue share, equity, barter, or unpaid arrangement, and whether acting directly
or through an intermediary entity. One person counts once even if they fill multiple roles.

1.10 “Gross Revenue” means total gross revenue in USD (or USD equivalent) for the relevant fiscal year,
aggregated across You and Your Affiliates.

1.11 “No Descriptive Locator Allowance” means that any required location reference in this License must be
a URL/URI only; descriptive phrases such as “see README,” “GitHub repo,” “Nature article,” “the folder,”
or other non-URI pointers do not satisfy the requirement.

──────────────────────────────────────────────────────────────────────────────
2. THE GRANT
──────────────────────────────────────────────────────────────────────────────

Subject to the terms and conditions of this License, the Licensor grants You a worldwide, royalty-free,
non-sublicensable, non-exclusive, irrevocable (for the term of copyright and similar rights) license to:

2.1 Share (NonCommercial): reproduce and Share the Sanctuary, in whole or in part, for NonCommercial purposes only.

2.2 Adapt (NonCommercial): produce, reproduce, and Share Adaptations for NonCommercial purposes only.

2.3 Technical modifications: make technical modifications necessary to exercise the granted rights
(e.g., format shifting or compatibility changes), provided such modifications do not circumvent or dilute
the License conditions.

2.4 No endorsement: This License does not grant any right to assert or imply endorsement by the Licensor.

2.5 Other rights excluded: No patent, trademark, privacy, or publicity rights are granted.

──────────────────────────────────────────────────────────────────────────────
3. CONDITIONS
──────────────────────────────────────────────────────────────────────────────

Your exercise of the licensed rights is expressly subject to all conditions below.

3.1 Attribution (human-readable)
When You Share the Sanctuary or an Adaptation, You must:
(a) credit the Licensor as: l’Evermoor;
(b) provide the License name and a copy of the License text or a link to it where reasonably practicable;
(c) indicate if changes were made (and describe changes reasonably);
(d) retain any notices supplied with the Sanctuary; and
(e) not imply endorsement.

3.2 Mandatory machine-readable Attribution JSON
When You Share the Sanctuary or an Adaptation, You must publish a machine-readable JSON record
(“Attribution JSON”) meeting all requirements below.

3.2.1 Filename and placement
Default filename: ATTRIBUTION.json
Placement: the root of the distribution, or if root placement is not applicable, embedded in an existing
manifest/metadata location that is reasonably accessible to recipients.

3.2.2 Required fields (Attribution JSON)
The Attribution JSON MUST include:
(a) "attribution_name": "l’Evermoor"
(b) "license": "ESL-ANCSA-MRA-IndiModSHA v1.0"
(c) "source_uri": a URL/URI where the Sanctuary (or the relevant shared portion) can be retrieved
    (No Descriptive Locator Allowance: URI only)
(d) "modified": boolean
(e) "modifications": string (may be empty if modified=false)
(f) "date_epoch": integer Unix epoch seconds
(g) "hash-prov-inh": a URL/URI to the corresponding Seal of Inherited Provenance entry (Section 5A)
    (URI only)

3.2.3 Serialization rule
All required JSON records under this License MUST be UTF-8.

3.3 NonCommercial restriction
You may not use the Sanctuary for Commercial purposes. This includes, without limitation:
(a) selling, licensing for a fee, or otherwise commercially distributing the Sanctuary; or
(b) selling, licensing for a fee, or otherwise commercially distributing any Adaptation; or
(c) bundling the Sanctuary (or any Adaptation) with a paid product/subscription/bonus or any gated access
whose primary purpose is commercial advantage or compensation.

3.4 ShareAlike for Adaptations
If You Share an Adaptation, You must license Your contributions under this same License, or a later
Licensor-published version that preserves Attribution + NonCommercial + ShareAlike with substantially
similar meaning and effect.

3.5 No downstream restrictions; no restrictive ETMs
You may not:
(a) impose additional or different terms restricting recipients’ exercise of rights granted by this License; or
(b) apply ETMs/DRM to the Sanctuary (or any Adaptation) if doing so restricts recipients from exercising
the licensed rights.

3.6 Platform-required ETMs (narrow allowance)
If a platform imposes ETMs as a technical necessity, You may distribute there only if:
(a) a functionally equivalent unencumbered (no-ETM) copy is also made available in parallel;
(b) at no additional charge; and
(c) with reasonable accessibility.

──────────────────────────────────────────────────────────────────────────────
4. INDEPENDENT MODULE SAFE HARBOR ALLOTMENT (IndiModSHA)
──────────────────────────────────────────────────────────────────────────────

4.1 Purpose
This Section creates a narrowly defined safe harbor permitting commercial distribution of truly Independent
Modules by Small Entities, while preserving the NonCommercial prohibition for the Sanctuary and all Adaptations.

4.2 Eligibility (“Small Entity”)
You qualify only if ALL of the following are true, aggregated across You and Your Affiliates:
(a) Module Production Team Headcount ≤ 19; AND
(b) Gross Revenue ≤ 247829 USD for the relevant fiscal year.

4.3 Permission granted (commercial, but limited)
If eligible, You MAY sell, license for a fee, or otherwise commercially distribute an Independent Module,
provided that:
(a) the Independent Module contains no portion of the Sanctuary (as defined in 1.7(a));
(b) the Independent Module is not an Adaptation of the Sanctuary; and
(c) the Independent Module is not marketed, represented, or packaged as including the Sanctuary.

4.4 Explicitly not permitted under the safe harbor
Even if eligible, You may NOT:
(a) commercially distribute the Sanctuary;
(b) commercially distribute any Adaptation;
(c) bundle the Sanctuary (or any Adaptation) with an Independent Module in a paid offering, including
“free with purchase,” “included as a bonus,” or gated access tied to payment; or
(d) provide paid access to the Sanctuary via hosting/SaaS/subscription/paywall/API/managed services where
the Sanctuary is part of what the customer receives.

4.5 Attribution still required
Use of this safe harbor does not waive Attribution duties when You Share the Sanctuary or an Adaptation.
Independent Modules that merely reference interoperability need not attribute, unless they include Sanctuary
content (which would disqualify them as Independent Modules).

4.6 Substantiation on request
Upon the Licensor’s request, You must provide reasonable substantiation of eligibility within 30 days.
Acceptable forms include an officer attestation and summary figures sufficient to substantiate eligibility.
No public disclosure is required.

4.7 Sunset on growth
If You cease to qualify as a Small Entity, permissions under this Section end automatically at the end of that
fiscal year. Continued commercial distribution thereafter violates this License. Past distributions while eligible
remain valid.

──────────────────────────────────────────────────────────────────────────────
5. DOWNSTREAM OFFER
──────────────────────────────────────────────────────────────────────────────

Every recipient of the Sanctuary automatically receives an offer from the Licensor to exercise the licensed rights
under this License, subject to its terms. You may not revoke or restrict this offer downstream.

──────────────────────────────────────────────────────────────────────────────
5A. SEAL OF INHERITED PROVENANCE (SIP)
──────────────────────────────────────────────────────────────────────────────

5A.1 Purpose
The Licensor maintains the Seal of Inherited Provenance: an append-only provenance chain recording descent,
forks, and Adaptations of the Sanctuary. Its function is provenance: a public memory of what begat what.

5A.2 When required
If You Share the Sanctuary or an Adaptation, You must publish a corresponding Provenance Seal at the time
of public release or as soon as reasonably practicable thereafter.

5A.3 Provenance Seal format (machine-readable)
Each Provenance Seal MUST be published as JSON and MUST include:
(a) "seal_id": a unique identifier (recommended: sha256 of the canonical JSON bytes);
(b) "attribution_name": "l’Evermoor";
(c) "license": "ESL-ANCSA-MRA-IndiModSHA v1.0";
(d) "date_epoch": integer Unix epoch seconds;
(e) "artifact": an object containing:
    - "name": release name or tag
    - "uri": a URL/URI where the released artifact can be retrieved (URI only)
    - "content_hash": a cryptographic digest of the artifact (recommended: sha256)
    - "hash_alg": the algorithm name (e.g., "sha256")
(f) "relation": one of ["original","share","adaptation","aggregation","independent_module_reference"];
(g) "ancestors": an array of zero or more parent identifiers and/or parent content hashes sufficient to trace
    ancestry (for Adaptations: MUST include at least one ancestor);
(h) "adapter": an object describing the adapter (You), minimally including a stable name or entity string;
(i) "signature": a detached signature over the canonical JSON payload excluding "signature" itself; and
(j) "sig_alg" identifying the scheme (recommended: ed25519).

5A.4 Canonicalization
The JSON used for hashing and signing MUST be deterministically serialized (“canonical JSON”):
keys sorted, UTF-8, no insignificant whitespace, and stable number formatting.

5A.5 Publication requirements
You must make each Provenance Seal publicly retrievable at a stable URL/URI, and include its URL/URI in the
Attribution JSON (Section 3.2) under the key:
"hash-prov-inh": "<URL/URI to the Provenance Seal>"  (URI only)

5A.6 External anchoring (optional)
You MAY additionally anchor the Provenance Seal by publishing its "seal_id" (or "content_hash") into an
external timestamping or blockchain system. If You do, You must include:
"anchor": { "system": "<name>", "tx_or_proof": "<identifier>", "anchored_hash": "<value>" }
Anchoring is permitted but not required; absence of anchoring does not waive obligations under this Section.

5A.7 No falsification
You may not knowingly publish Provenance Seals that materially misrepresent ancestry, authorship, or content
hashes. Honest mistakes cured promptly do not constitute a breach.

──────────────────────────────────────────────────────────────────────────────
6. DISCLAIMER AND LIMITATION OF LIABILITY
──────────────────────────────────────────────────────────────────────────────

6.1 Disclaimer of warranties
THE SANCTUARY IS PROVIDED “AS IS” AND “AS AVAILABLE,” WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ACCURACY, OR ABSENCE OF DEFECTS.

6.2 Limitation of liability
To the extent permitted by law, the Licensor is not liable on any legal theory for any direct, special, indirect,
incidental, consequential, punitive, or exemplary damages arising out of this License or use of the Sanctuary,
even if advised of the possibility.

──────────────────────────────────────────────────────────────────────────────
7. TERM, TERMINATION, REINSTATEMENT
──────────────────────────────────────────────────────────────────────────────

7.1 Term
This License applies for the term of copyright and similar rights.

7.2 Automatic termination
Rights terminate automatically upon any failure to comply with this License.

7.3 Reinstatement upon cure
If You cure the violation within 30 days of discovery, Your rights reinstate automatically as of the cure date,
unless the Licensor terminated Your rights in writing before cure. Reinstatement does not erase liability for
violations prior to cure.

──────────────────────────────────────────────────────────────────────────────
8. MISCELLANEOUS
──────────────────────────────────────────────────────────────────────────────

8.1 Severability
If any provision is unenforceable, it is reformed to the minimum extent necessary to be enforceable; if not
possible, it is severed without affecting the remaining provisions.

8.2 No waiver
Failure to enforce any term is not a waiver.

8.3 Interpretation
This License does not reduce or restrict uses permitted without permission under applicable law (e.g., fair use/fair dealing).

Canonical constants:
- License ID: ESL-ANCSA-MRA-IndiModSHA
- Revenue cap (USD): 247829
- Headcount cap (people): 19
- Attribution name: l’Evermoor
- Machine key for SIP URI: hash-prov-inh
