// Halaman deteksi: pilih/seret file + pratinjau, kamera (getUserMedia), status kirim.
// Kamera: frame video -> canvas (crop persegi tengah = yang dilihat model) -> JPEG -> input file -> form dikirim seperti upload.
// getUserMedia hanya jalan di HTTPS atau localhost (PythonAnywhere = HTTPS).
(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("form"), input = $("image"), drop = $("drop"), preview = $("drop-preview"), name = $("drop-name");
  const cam = $("camera"), video = $("video"), err = $("cam-error"), submit = $("submit");
  let stream = null, previewUrl = null;

  const showFile = (file) => {
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    name.textContent = file.name;
    name.hidden = false;
    drop.classList.add("has-file");
  };
  input.addEventListener("change", () => showFile(input.files[0]));

  // seret & lepas
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("is-over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f || typeof DataTransfer === "undefined") return;
    const dt = new DataTransfer();
    dt.items.add(f);
    input.files = dt.files;
    showFile(f);
  });

  // status kirim: cegah klik ganda, beri umpan balik
  form.addEventListener("submit", () => {
    submit.disabled = true;
    submit.querySelector("span").textContent = "Menganalisis…";
  });
  window.addEventListener("pageshow", () => {  // kembali via tombol back: pulihkan tombol
    submit.disabled = false;
    submit.querySelector("span").textContent = "Analisis daun";
  });

  // hasil baru: di layar sempit gulir ke panel hasil
  if (document.querySelector(".result") && window.matchMedia("(max-width: 860px)").matches) {
    $("hasil").scrollIntoView({ block: "start" });
  }

  // kamera
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof DataTransfer === "undefined") return;
  $("cam-open").hidden = false;

  const stop = () => { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; cam.hidden = true; };
  const fail = (msg) => { err.textContent = msg; err.hidden = false; stop(); };

  $("cam-open").onclick = async () => {
    err.hidden = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream;
      cam.hidden = false;
      cam.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
      fail(e.name === "NotAllowedError" ? "Izin kamera ditolak. Izinkan akses kamera di pengaturan browser."
                                        : "Kamera tidak bisa dibuka (" + e.name + "). Gunakan upload file.");
    }
  };
  $("cam-close").onclick = stop;
  video.addEventListener("loadedmetadata", () => {  // kotak panduan = crop persegi tengah, potret maupun lanskap
    const w = video.videoWidth, h = video.videoHeight, s = Math.min(w, h), g = document.querySelector(".guide");
    g.style.width = (100 * s / w) + "%";
    g.style.height = (100 * s / h) + "%";
  });

  $("cam-shot").onclick = () => {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return fail("Kamera belum siap, coba lagi.");
    const s = Math.min(w, h);
    const c = document.createElement("canvas");
    c.width = c.height = Math.min(s, 1024);
    c.getContext("2d").drawImage(video, (w - s) / 2, (h - s) / 2, s, s, 0, 0, c.width, c.height);
    c.toBlob((blob) => {
      if (!blob) return fail("Gagal mengambil foto.");
      const dt = new DataTransfer();
      dt.items.add(new File([blob], "kamera.jpg", { type: "image/jpeg" }));
      input.files = dt.files;
      showFile(dt.files[0]);
      stop();
      form.requestSubmit ? form.requestSubmit() : form.submit();
    }, "image/jpeg", 0.92);
  };
  window.addEventListener("pagehide", stop);
})();
