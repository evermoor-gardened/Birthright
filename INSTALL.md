# Installing ProvidenceOS

ProvidenceOS is designed to install on **any device** without modification. The installation process is the same whether you are using a 30‑year‑old PC, a modern smartphone, or an embedded sensor.

## Prerequisites
- A device capable of reading the installation medium (USB, optical disc, network boot, or direct flash).
- No other requirements. The system handles everything.

## Installation Steps

1. **Obtain the Installation Image**
   Download the ProvidenceOS ISO or disk image from a trusted source. Verify its cryptographic signature (provided separately) to ensure integrity.

2. **Write the Image to Media**
   Use any standard tool (`dd`, `balenaEtcher`, `Rufus`) to write the image to a USB drive, SD card, or other bootable media.

3. **Boot from the Media**
   Insert the media into your target device and boot from it. You may need to adjust BIOS/UEFI settings to allow booting from external media.

4. **Watch the Weaver**
   Upon boot, you will see a black screen with the message:
   > "You are home. Rest now. All is well."

   Then, a string of scattered, colored lights (representing your hardware components) appears. Watch as they gently drift together, forming a single, coherent necklace. This is the Universal Driver Weaver handshaking with your device.

5. **The Desktop Appears**
   When the necklace is complete, the screen shifts to a deep indigo. The paired icon of the open gear and crystalline lock fades in, pulsing softly. You are now in the Sanctuary.

6. **No Account Creation**
   There is no login, no password, no tracking. The system simply is. Your identity is derived from the hardware itself (encrypted and stored in the Private Kernel). If you wish to transfer your identity to another device, use the "Embrace" protocol to pair them.

## Post‑Installation
- Explore the "Rooms" (filesystem) by clicking the Velvet Curtain icon.
- Connect with other devices via "The Embrace" (messenger icon).
- Adjust permissions via "The Hairpin" (consent manager).
- The system will continue to learn and adapt to your patterns.

## Running the Reference Kernel

The Python reference implementation can be run standalone for development and testing:

```bash
# Install dependencies
pip install requests numpy

# Optional: Install Ollama for local LLM inference
# See https://ollama.ai for installation

# Launch ProvidenceOS Reference Kernel
python providence_kernel.py

# Launch in headless/CLI mode
python providence_kernel.py --headless

# Run verification suite
python providence_kernel.py --verify
```

## Troubleshooting
If the Weaver fails to complete the necklace, it means the hardware introspection encountered an unknown component. This is rare; the system will display a diagnostic pearl and ask if you want to attempt a fallback generic driver. In most cases, this resolves the issue.
