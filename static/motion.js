// Motion (GSAP 3.13 + ScrollTrigger + SplitText). Semua animasi dimatikan untuk prefers-reduced-motion.
// Tanpa GSAP (CDN gagal/JS mati): kelas "js" dilepas -> semua konten tampil statis.
(() => {
  const root = document.documentElement;
  if (!window.gsap || !window.ScrollTrigger) { root.classList.remove("js"); return; }
  gsap.registerPlugin(ScrollTrigger);
  const hasSplit = !!window.SplitText;
  if (hasSplit) gsap.registerPlugin(SplitText);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");

  if (reduce.matches) {           // tampil statis, tanpa tirai
    root.classList.remove("js");
    window.__motion = true;
    return;
  }

  // tirai masuk halaman
  gsap.to(".curtain", { scaleY: 0, transformOrigin: "50% 0%", duration: 0.9, ease: "expo.inOut", delay: 0.05 });

  // judul: muncul per huruf
  document.querySelectorAll("[data-split]").forEach((h, i) => {
    gsap.set(h, { opacity: 1 });
    if (!hasSplit) return gsap.from(h, { y: 30, opacity: 0, duration: 1, ease: "expo.out", delay: 0.4 });
    const s = new SplitText(h, { type: "words,chars", wordsClass: "split-word" });
    const inHero = h.closest(".hero, .page-hero");
    gsap.from(s.chars, {
      yPercent: 115, rotate: 8, opacity: 0, duration: 0.9, stagger: 0.016, ease: "expo.out",
      delay: inHero ? 0.45 : 0,
      scrollTrigger: inHero ? null : { trigger: h, start: "top 85%" },
      onComplete: () => s.revert(),  // kembalikan teks asli (pembaca layar, copy-paste)
    });
  });

  // elemen hero berurutan
  gsap.to("[data-hero]", { opacity: 1, y: 0, duration: 1, stagger: 0.12, ease: "expo.out", delay: 0.75 });

  // ilustrasi daun: helai membesar, urat tergambar
  const blade = document.querySelector(".leaf-art .blade");
  if (blade) {
    gsap.from(".leaf-art svg", { scale: 0.86, rotate: -10, opacity: 0, transformOrigin: "50% 90%", duration: 1.5, ease: "expo.out", delay: 0.3 });
    document.querySelectorAll(".leaf-art .vein").forEach((p, i) => {
      const L = p.getTotalLength();
      gsap.fromTo(p, { strokeDasharray: L, strokeDashoffset: L }, { strokeDashoffset: 0, duration: 1.4, ease: "power2.inOut", delay: 0.7 + i * 0.08 });
    });
    gsap.from(".chip", { y: 16, opacity: 0, scale: 0.9, duration: 0.8, stagger: 0.25, ease: "back.out(1.7)", delay: 1.6 });
    gsap.to(".leaf-art", { yPercent: 12, ease: "none", scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true } });
  }

  // daun melayang: parallax beda kecepatan (goyang ditangani CSS)
  gsap.utils.toArray(".float-leaf").forEach((l, i) => {
    gsap.to(l, { yPercent: -(80 + i * 45), ease: "none", scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true } });
  });

  // muncul saat di-scroll (konten)
  gsap.utils.toArray("[data-reveal]").forEach((el) => {
    gsap.to(el, { opacity: 1, y: 0, duration: 0.9, ease: "expo.out", scrollTrigger: { trigger: el, start: "top 88%" } });
  });

  // garis urat penghubung langkah: tergambar mengikuti scroll
  const line = document.querySelector(".steps-line path");
  if (line) {
    const L = line.getTotalLength();
    gsap.fromTo(line, { strokeDasharray: L, strokeDashoffset: L }, {
      strokeDashoffset: 0, ease: "none", scrollTrigger: { trigger: ".steps-grid", start: "top 80%", end: "bottom 60%", scrub: true },
    });
  }

  // angka statistik menghitung naik
  document.querySelectorAll("[data-count]").forEach((el) => {
    const end = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), o = { v: 0 };
    const fmt = (v) => v.toLocaleString("id-ID", { minimumFractionDigits: dec, maximumFractionDigits: dec });
    el.textContent = fmt(0);
    gsap.to(o, { v: end, duration: 1.8, ease: "power2.out", scrollTrigger: { trigger: el, start: "top 90%" }, onUpdate: () => (el.textContent = fmt(o.v)) });
  });

  // hasil deteksi: bar probabilitas tumbuh
  gsap.from(".bar i", { scaleX: 0, duration: 1.2, ease: "expo.out", stagger: 0.12, delay: 0.6 });
  gsap.from(".result-img", { scale: 0.9, opacity: 0, duration: 0.9, ease: "expo.out", delay: 0.4 });

  // tirai keluar saat pindah halaman internal
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || a.target || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const u = new URL(a.href, location.href);
    if (u.origin !== location.origin || (u.pathname === location.pathname && u.hash)) return;  // anchor di halaman sama: scroll biasa
    e.preventDefault();
    gsap.to(".curtain", { scaleY: 1, transformOrigin: "50% 100%", duration: 0.6, ease: "expo.in", onComplete: () => (location.href = u.href) });
  });
  window.addEventListener("pageshow", (e) => { if (e.persisted) gsap.set(".curtain", { scaleY: 0 }); });  // kembali dari bfcache

  window.__motion = true;
})();
