/**
 * One-click demo texts.
 *
 * Curated to cover the range a demo needs: an obvious scam in each language, a
 * clean legitimate posting in each, and the borderline case that shows the system
 * is graded rather than binary.
 *
 * The Indonesian scam is the concept paper's own worked scenario (section 3.4).
 */

export interface Example {
  id: string;
  label: string;
  locale: "en" | "id";
  expectation: "scam" | "legit" | "borderline";
  text: string;
}

export const EXAMPLES: Example[] = [
  {
    id: "id-scam",
    label: "Scam (Indonesian)",
    locale: "id",
    expectation: "scam",
    text: `LOWONGAN KERJA ADMIN ONLINE
Dibutuhkan segera admin online untuk perusahaan ternama.
Gaji Rp9.000.000 per bulan, tanpa pengalaman, langsung kerja dari rumah.
Kuota terbatas hanya untuk 10 orang pertama!
Wajib membayar biaya administrasi sebesar Rp250.000 untuk proses berkas.
Interview dilakukan via Telegram.
Kirim CV dan foto KTP ke hrd.rekrutmen2024@gmail.com`,
  },
  {
    id: "en-scam",
    label: "Scam (English)",
    locale: "en",
    expectation: "scam",
    text: `URGENT HIRING - Data Entry Clerk (Work From Home)
Earn $5,000 per month, no experience necessary, start immediately!
Limited slots available - act now before they are gone.
A one-time administration fee of $50 is required to process your application.
Interview conducted via Telegram.
Send your CV and a photo of your ID to hiring.dept2024@gmail.com`,
  },
  {
    id: "id-legit",
    label: "Legitimate (Indonesian)",
    locale: "id",
    expectation: "legit",
    text: `Software Engineer (Backend) - PT Teknologi Nusantara
Lokasi: Yogyakarta, Indonesia. Tipe: Full-time, hybrid.
Kualifikasi: S1 Ilmu Komputer atau setara, pengalaman minimal 2 tahun membangun layanan backend dengan Python atau Go, memahami PostgreSQL dan sistem terdistribusi.
Tanggung jawab: merancang dan memelihara layanan API internal, melakukan code review, berkolaborasi dengan tim produk.
Rentang gaji: Rp12.000.000 - Rp18.000.000 per bulan sesuai pengalaman.
Lamaran dikirim melalui halaman karier resmi kami di https://karier.teknologinusantara.co.id`,
  },
  {
    id: "en-legit",
    label: "Legitimate (English)",
    locale: "en",
    expectation: "legit",
    text: `Senior Backend Engineer - Acme Technologies
Location: Seattle, Washington. Full-time, hybrid.
Requirements: BS in Computer Science or equivalent, at least 4 years of experience building backend services in Python or Go, strong PostgreSQL knowledge.
Responsibilities: design and maintain internal API services, conduct code reviews, collaborate with product teams.
Salary range: $9,000 - $12,000 per month depending on experience.
Apply through our careers page at https://careers.acmetechnologies.com`,
  },
  {
    id: "id-borderline",
    label: "Borderline (Indonesian)",
    locale: "id",
    expectation: "borderline",
    text: `Dicari Admin Media Sosial untuk UMKM Batik Yogya.
Kerja part time, 4 jam per hari, gaji Rp2.000.000 per bulan.
Butuh yang bisa desain Canva dan paham Instagram.
Kirim portofolio ke batikyogya.umkm@gmail.com atau WhatsApp 081234567890.`,
  },
  {
    id: "en-borderline",
    label: "Borderline (English)",
    locale: "en",
    expectation: "borderline",
    text: `Social Media Assistant needed for a small handmade jewellery business.
Part time, work from home, roughly 10 hours a week.
We are looking for someone who knows Instagram and can edit simple graphics.
Pay is $600 per month to start, reviewed after three months.
Email your portfolio to studio.jewellery.hiring@gmail.com`,
  },
];
