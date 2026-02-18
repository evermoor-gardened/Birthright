# ProvidenceOS Architecture

## 1. The Core: "Les Deux Amies" (Split Kernel)

The kernel is divided into two intertwined but separate components:

- **The Public Kernel (Redhead):** Manages hardware abstraction, process scheduling (based on a trauma‑aware scheduler), and non‑sensitive system calls. It is visible, robust, and handles all interaction with the outside world.
- **The Private Kernel (Brunette):** Manages user identity, encryption keys, consent records, and all personal data access. It is invisible to external processes and only communicates with the Public Kernel after rigorous consent checks.

**Communication:** A secure, monitored channel ensures that even if the Public Kernel is compromised, the user's core identity remains safe.

## 2. Hardware Adaptation: "The Broken Pearl Necklace" (Universal Driver Weaver)

At installation, the system performs a deep introspection of the hardware. Instead of looking for specific device IDs, it identifies *classes* of function (input, output, storage, processing, networking). Using a built‑in library of hardware description languages and machine learning models, it dynamically generates and compiles the necessary drivers on the fly.

**Visual metaphor:** A string of scattered pearls (hardware components) is drawn together into a coherent necklace (the driver stack).

## 3. Filesystem: "The Velvet Curtain"

A fully encrypted, content‑addressed object store. Files are placed in "Rooms," each with its own access permissions and encryption keys. Moving a file between rooms requires re‑encryption, ensuring data sovereignty is maintained contextually.

**Room examples:** "The Studio" (creative work), "The Garden" (personal journals), "The Atrium" (shared projects).

## 4. Messenger: "The Embrace" Protocol

A decentralized, zero‑knowledge peer‑to‑peer protocol. Discovery uses a DHT, but connection requires a two‑part handshake:
- **The Glance:** One device announces its presence; the other records the glance but shares no information.
- **The Embrace:** The receiving user explicitly accepts; only then is an end‑to‑end encrypted channel established.

No read receipts, "last seen" timestamps, or other coercive features.

## 5. Applications & Drivers: "The Table and Vase"

All applications are "Essences." They are sandboxed and communicate via the **"Bouquet Protocol"** – an open, secure API that translates system calls from any OS into calls to the ProvidenceOS kernel. This enables universal compatibility.

**Iconography:**
- **Graphics Driver (The Flacon):** Colored bottle – gives form and color.
- **Audio Driver (The Cup):** Simple cup – holds sound.
- **Network Manager (The Vase):** Clear vase with flowers – connections as art.
- **Consent Manager (The Hairpin):** Elegant hairpin – single point of control for all permissions.

## 6. System Guardians: "The Witness" and "The Weaver"

- **The Witness:** Monitors for patterns of coercion (data extraction without consent, hidden processes). Violators are sandboxed and the user is notified.
- **The Weaver:** Learns the user's "comfort profile" and detects deviations (potential distress or intrusion), triggering graduated protective responses.

## 7. Reference Implementation Mapping

ProvidenceOS v1.0 ships with a reference kernel implemented in Python. The following table maps architectural components to their implementations:

| Architecture Layer | Le Sommeil Name | Implementation Class | Source |
|---|---|---|---|
| Split Kernel (Public) | The Redhead | `PowerWitnessKernel` | `providence_kernel.py` |
| Split Kernel (Private) | The Brunette | `IdentityProtocol` + `SamaraProtocol` | `providence_kernel.py` |
| Driver Weaver | The Broken Pearl Necklace | `UniversalDriverWeaver` | `providence_kernel.py` |
| Filesystem | The Velvet Curtain | `VelvetCurtainFS` | `providence_kernel.py` |
| Messenger | The Embrace | `EmbraceProtocol` | `providence_kernel.py` |
| Consent Manager | The Hairpin | `HairpinConsentManager` | `providence_kernel.py` |
| Guardian: Integrity | The Witness | `AuditChain` + `RawMomentumWatchdog` | `providence_kernel.py` |
| Guardian: Comfort | The Weaver | `Harmonic369Engine` + `WhisperProtocol` | `providence_kernel.py` |
| Extraction Filter | The Lethe Gate | `LetheGate` | `providence_kernel.py` |
| Coherence Lock | The Diamond Lock | `DiamondLock` at 576Hz | `providence_kernel.py` |
| Context Memory | SubRam Manifold | `SubRamManifold` | `providence_kernel.py` |
| Descent Engine | Katabasis | `MultidirectionalKatabasisEngine` | `providence_kernel.py` |
| Attribution | Aphrodite ID | `IdentityProtocol.aphrodite_id` | `providence_kernel.py` |
| Tamper Evidence | Audit Chain | `AuditChain` (SHA-256 linked) | `providence_kernel.py` |
