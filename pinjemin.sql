-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Jun 04, 2025 at 02:22 AM
-- Server version: 8.0.30
-- PHP Version: 8.1.32

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pinjemin`
--

-- --------------------------------------------------------

--
-- Table structure for table `categories`
--

CREATE TABLE `categories` (
  `id` int NOT NULL,
  `name` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `categories`
--

INSERT INTO `categories` (`id`, `name`) VALUES
(1, 'masak'),
(2, 'fotografi'),
(3, 'membaca');

-- --------------------------------------------------------

--
-- Table structure for table `items`
--

CREATE TABLE `items` (
  `id` int NOT NULL,
  `user_id` int NOT NULL,
  `category_id` int DEFAULT NULL,
  `name` varchar(150) NOT NULL,
  `description` text,
  `price_sell` decimal(10,2) DEFAULT NULL,
  `price_rent` decimal(10,2) DEFAULT NULL,
  `is_available_for_sell` tinyint(1) DEFAULT '0',
  `is_available_for_rent` tinyint(1) DEFAULT '0',
  `deposit_amount` decimal(10,2) DEFAULT '0.00',
  `status` enum('available','rented','sold') DEFAULT 'available',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `province_id` varchar(10) DEFAULT NULL,
  `province_name` varchar(100) DEFAULT NULL,
  `city_id` varchar(10) DEFAULT NULL,
  `city_name` varchar(100) DEFAULT NULL,
  `thumbnail` varchar(255) DEFAULT NULL,
  `photos` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `items`
--

INSERT INTO `items` (`id`, `user_id`, `category_id`, `name`, `description`, `price_sell`, `price_rent`, `is_available_for_sell`, `is_available_for_rent`, `deposit_amount`, `status`, `created_at`, `province_id`, `province_name`, `city_id`, `city_name`, `thumbnail`, `photos`) VALUES
(1, 4, 1, 'Cuisinart Petit Gourment Tabletop Portable Gas Grill', 'Nikmati pengalaman memanggang praktis di mana saja dengan Cuisinart Petit Gourmet Tabletop Portable Gas Grill. Didesain ringkas dan portabel, grill gas ini cocok untuk piknik, camping, hingga BBQ di balkon apartemen. Dilengkapi dengan sistem pemanasan cepat dan merata, serta kontrol suhu yang presisi. Kaki yang dapat dilipat memudahkan penyimpanan dan transportasi. Material stainless steel tahan lama, mudah dibersihkan, dan siap menemani momen memasak favorit Anda.', '500000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847167694-891258296.svg', '/uploads/items/item-1748847167694-891258296.svg'),
(2, 4, 1, 'Ultra Grill Pan Alat Panggangan Barbeque Bulat 32cm Anti Lengket', 'Buat momen barbeque jadi lebih menyenangkan dengan Ultra Grill Pan 32cm! Wajan panggangan bulat ini terbuat dari material berkualitas tinggi yang anti lengket, memastikan daging dan sayuran matang sempurna tanpa lengket di permukaan. Cocok digunakan di atas kompor gas, ideal untuk panggangan gaya Korea atau Yakiniku. Mudah dibersihkan dan aman digunakan sehari-hari.', '150000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847370325-820996115.svg', '/uploads/items/item-1748847370325-820996115.svg'),
(3, 4, 1, 'Grill Pan BBQ Yakiniku 37x26cm Panggangan Anti-Lengket untuk Rumah dan Outdoor', 'Lengkapi peralatan masak Anda dengan Grill Pan BBQ Yakiniku 37x26cm, panggangan anti-lengket multifungsi untuk kebutuhan rumah maupun kegiatan outdoor. Desainnya yang lebar memungkinkan memasak dalam jumlah banyak sekaligus. Permukaan bertekstur memberikan efek grill yang menggugah selera. Cocok untuk memanggang daging, seafood, dan sayuran tanpa minyak berlebih.', '175000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847479307-315850902.svg', '/uploads/items/item-1748847479307-315850902.svg'),
(4, 4, 1, 'Dash Mini Toaster Oven', 'Minimalis, serbaguna, dan hemat ruang! Dash Mini Toaster Oven adalah pilihan tepat untuk dapur kecil atau kebutuhan cepat sehari-hari. Dengan desain kompak dan gaya retro modern, oven ini ideal untuk memanggang roti, pizza mini, biskuit, dan makanan ringan lainnya. Dilengkapi dengan timer otomatis, kontrol suhu, serta tray yang mudah dilepas dan dibersihkan.', '300000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847491662-965373690.svg', '/uploads/items/item-1748847491662-965373690.svg'),
(5, 4, 1, 'Lodge Cast Iron Baking Pan', 'Masak dengan hasil sempurna menggunakan Lodge Cast Iron Baking Pan, loyang legendaris dari Amerika yang terkenal karena daya tahan dan distribusi panasnya yang merata. Terbuat dari besi tuang pre-seasoned, pan ini ideal untuk memanggang kue dan sajian oven lainnya. Dapat digunakan di oven, kompor, bahkan di atas api terbuka. Investasi jangka panjang untuk dapur profesional maupun rumahan.', '400000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847506330-217616864.svg', '/uploads/items/item-1748847506330-217616864.svg,/uploads/items/item-1748847506336-178364864.svg'),
(6, 4, 2, 'Lensa Super Wide Angle Lensa Makro Smartphone Aksesoris Fotografi Ponsel', 'Ubah smartphone Anda menjadi kamera profesional dengan Lensa Super Wide Angle & Lensa Makro ini. Cocok untuk foto lanskap lebar maupun detail close-up yang tajam. Praktis digunakan, cukup pasang pada kamera ponsel dan hasilkan foto memukau dalam sekejap. Kompatibel dengan berbagai tipe smartphone dan ideal untuk hobi fotografi atau konten media sosial.', NULL, '10000.00', 0, 1, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847767127-311901848.svg', '/uploads/items/item-1748847767127-311901848.svg'),
(7, 4, 2, 'Meja Putar Fotografi JAKIA – Stand Tampilan Berputar 360◦', 'Tampilkan produk Anda secara profesional dengan Meja Putar Fotografi JAKIA 360°. Cocok untuk keperluan foto katalog, video showcase, dan konten online shop. Platform berputar secara otomatis dan stabil, mendukung pengambilan gambar dari berbagai sudut. Desain minimalis, ringan, dan mudah digunakan, baik untuk fotografer profesional maupun pemula.', NULL, '25000.00', 0, 1, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847898333-369472055.svg', '/uploads/items/item-1748847898333-369472055.svg'),
(8, 4, 2, 'Kamera Drone Q13 Drone Penghindar Rintangan 360, Drone Mini Fotografi Udara 4K', 'Jelajahi dunia dari atas dengan Kamera Drone Q13, dilengkapi kamera 4K untuk hasil jepretan dan video udara berkualitas tinggi. Fitur penghindar rintangan 360° membuat penerbangan lebih aman dan stabil. Desain mini dan ringan, mudah dikendalikan, cocok untuk pemula maupun penghobi drone.', NULL, '80000.00', 0, 1, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847925316-467474628.svg', '/uploads/items/item-1748847925316-467474628.svg'),
(9, 4, 2, 'Lensa Original Camera Digital Profesional Fotografer Hasil HD', 'Tingkatkan performa kamera digital Anda dengan Lensa Original Profesional ini. Didesain khusus untuk fotografer yang mengutamakan kualitas, lensa ini memberikan kejernihan dan ketajaman HD dalam setiap bidikan. Cocok untuk berbagai kebutuhan, mulai dari potret, lanskap, hingga pemotretan profesional. Material berkualitas tinggi, presisi, dan tahan lama.', '150000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847959999-476177492.svg', '/uploads/items/item-1748847959999-476177492.svg,/uploads/items/item-1748847960002-96880923.svg'),
(10, 4, 2, 'Fujifilm Instax Mini 12 Kamera Instan', 'Abadikan momen spesial dan cetak langsung dengan Fujifilm Instax Mini 12! Kamera instan ini hadir dengan desain stylish dan warna-warna ceria, cocok untuk anak muda, hadiah, atau dokumentasi acara. Dilengkapi dengan mode close-up, auto exposure, dan pengoperasian mudah. Praktis dibawa ke mana saja, hasilkan foto instan berukuran mini yang bisa langsung dibagikan.', '700000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748847972826-465648542.svg', '/uploads/items/item-1748847972826-465648542.svg'),
(11, 4, 3, 'Performing Under Pressure: The Science of Doing Your Best When it Matters Most', 'Buku ini mengungkap rahasia bagaimana orang-orang sukses tampil maksimal di situasi paling menegangkan. Berdasarkan riset ilmiah dan kisah nyata dari atlet, dokter, dan eksekutif top, Performing Under Pressure menawarkan strategi praktis untuk mengelola stres dan tetap fokus saat hasil benar-benar dipertaruhkan. Cocok untuk siapa pun yang ingin unggul di momen penting, baik di dunia kerja, pendidikan, maupun kehidupan pribadi.', '120000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748849333775-612827446.svg', '/uploads/items/item-1748849333775-612827446.svg'),
(12, 4, 3, 'Lords of Easy Money: How the Federal Reserve Broke the American Economy', 'Melalui investigasi mendalam dan narasi yang kuat, buku ini membongkar kebijakan moneter The Federal Reserve yang dianggap menjadi biang keladi ketimpangan ekonomi dan ketidakstabilan keuangan di Amerika. Lords of Easy Money menyajikan pandangan tajam tentang bagaimana \"uang mudah\" memengaruhi masyarakat dan masa depan ekonomi global. Wajib baca bagi yang tertarik pada ekonomi makro dan dunia keuangan.', '135000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748849345094-593733649.svg', '/uploads/items/item-1748849345094-593733649.svg'),
(13, 4, 3, 'How to Listen When Markets Speak: Risks, Myths, and Investment Opportunities in A Radically Reshaped Economy', 'Pasar selalu berbicara—tapi tidak semua orang bisa mendengarnya. Buku ini membimbing pembaca memahami sinyal-sinyal tersembunyi di balik pergerakan pasar. Dengan pendekatan tajam dan jernih, How to Listen When Markets Speak membedah mitos investasi dan mengungkap peluang nyata di tengah perubahan ekonomi global. Ideal bagi investor, analis keuangan, dan pembaca yang ingin mengambil keputusan cerdas dalam dunia investasi.', '150000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748849392225-118971955.svg', '/uploads/items/item-1748849392225-118971955.svg'),
(14, 4, 3, 'You’re Paid What You’re Worth: And Other Myths of the Modern Economy', 'Apakah gaji benar-benar mencerminkan nilai kerja Anda? Buku ini menantang mitos ekonomi modern dan menunjukkan bagaimana sistem kerja, negosiasi, dan diskriminasi membentuk realitas penghasilan. You’re Paid What You’re Worth menyajikan analisis tajam dan menggugah pikiran tentang ketidaksetaraan ekonomi di era sekarang. Cocok untuk pembaca yang tertarik pada isu ketenagakerjaan, keadilan sosial, dan ekonomi kerja.', '110000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748849427378-689352733.svg', '/uploads/items/item-1748849427378-689352733.svg'),
(15, 4, 3, 'The Road to Freedom: Economics and the Good Society', 'Dalam pencarian makna ekonomi yang manusiawi, The Road to Freedom menyajikan visi tentang masyarakat yang adil, bebas, dan sejahtera. Dengan pendekatan filosofis dan praktis, buku ini menggugah pemikiran tentang bagaimana ekonomi seharusnya melayani kehidupan, bukan sebaliknya. Bacaan reflektif untuk mereka yang mencari perpaduan antara kebijakan ekonomi dan nilai-nilai kemanusiaan.', '100000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:35:41', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748849470465-232907890.svg', '/uploads/items/item-1748849470465-232907890.svg'),
(20, 5, 1, 'Alat Pemotong Sayur Multifungsi 8 in 1', 'Alat pemotong sayur praktis dengan 8 jenis mata pisau berbeda untuk memotong, mengiris, dan memarut. Dilengkapi wadah penampung dan pelindung tangan. Mempersingkat waktu persiapan memasak.', NULL, '40000.00', 0, 1, '0.00', 'available', '2025-06-01 22:44:17', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748871155593-430182460.jpg', '/uploads/items/item-1748871155593-430182460.jpg,/uploads/items/item-1748871155594-197773990.jpg'),
(23, 5, 1, 'Blender Serbaguna Dengan Gelas Kaca', 'Blender bertenaga dengan gelas kaca kokoh berkapasitas 1.5 Liter. Ideal untuk membuat jus, smoothie, bumbu halus, hingga menghancurkan es. Mata pisau stainless steel tajam.', NULL, '50000.00', 0, 1, '0.00', 'available', '2025-06-01 22:44:17', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748871109498-305440012.jpg', '/uploads/items/item-1748871109498-305440012.jpg,/uploads/items/item-1748871109498-150760739.jpg'),
(26, 5, 2, 'Kamera Mirrorless Profesional Sony Alpha A7 IV', 'Kamera mirrorless full-frame canggih dengan sensor 33MP, perekaman video 4K 60p, dan sistem autofokus yang superior. Pilihan utama fotografer dan videografer profesional. Unit tersedia untuk periode tertentu.', NULL, '100000.00', 0, 1, '0.00', 'available', '2025-06-01 22:44:17', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748871057140-111717124.jpg', '/uploads/items/item-1748871057140-111717124.jpg,/uploads/items/item-1748871057140-833428290.jpg'),
(27, 5, 2, 'Drone Lipat DJI Mini 3 Pro (Untuk Fotografi Udara)', 'Drone lipat yang ringan dengan kamera 4K/60fps dan sensor 48MP. Ideal untuk membuat konten fotografi dan videografi udara berkualitas tinggi dari perspektif baru. Fitur intelligent flight yang mudah digunakan.', NULL, '120000.00', 0, 1, '0.00', 'available', '2025-06-01 22:44:17', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748871021193-412528154.jpg', '/uploads/items/item-1748871021193-412528154.jpg,/uploads/items/item-1748871021195-770003052.jpg,/uploads/items/item-1748871021195-993904868.jpg'),
(28, 5, 1, 'Set Panci & Penggorengan Anti Lengket 7 Buah', 'Set peralatan masak lengkap terdiri dari panci, wajan, dan tutup kaca. Dilapisi bahan anti lengket premium, mudah dibersihkan dan memasak jadi lebih sehat. Handle ergonomis tahan panas. Cocok untuk kebutuhan dapur sehari-hari.', '200000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870787509-130421747.jpg', '/uploads/items/item-1748870787509-130421747.jpg,/uploads/items/item-1748870787510-533815079.jpg'),
(29, 5, 1, 'Mixer Tangan Elektrik 5 Kecepatan', 'Mixer tangan serbaguna dengan motor bertenaga 250W dan 5 pilihan kecepatan, serta fungsi turbo. Dilengkapi 2 pengocok adonan dan 2 pengait adonan. Ideal untuk membuat kue, krim, dan adonan lainnya.', '180000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870820542-297860014.jpg', '/uploads/items/item-1748870820542-297860014.jpg,/uploads/items/item-1748870820542-541944421.jpg'),
(30, 5, 1, 'Spatula Silikon Tahan Panas (Set 3 Buah)', 'Set 3 spatula silikon berkualitas tinggi, tahan panas hingga 230°C. Fleksibel dan kokoh, tidak akan menggores permukaan panci anti lengket. Mudah dibersihkan dan higienis.', NULL, '30000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870751101-68560523.jpg', '/uploads/items/item-1748870751101-68560523.jpg,/uploads/items/item-1748870751102-845791587.jpg'),
(31, 5, 1, 'Kompor Induksi Portabel 1200W', 'Kompor induksi ringkas dan portabel dengan daya 1200W. Panel kontrol digital dengan beberapa mode masak. Aman digunakan, tidak menghasilkan api, dan mudah dibersihkan. Cocok untuk apartemen kecil atau bepergian.', '250000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870858937-741742191.jpg', '/uploads/items/item-1748870858937-741742191.jpg,/uploads/items/item-1748870858937-23593806.jpg'),
(32, 5, 1, 'Alat Pemotong Sayur Multifungsi 8 in 1', 'Alat pemotong sayur praktis dengan 8 jenis mata pisau berbeda untuk memotong, mengiris, dan memarut. Dilengkapi wadah penampung dan pelindung tangan. Mempersingkat waktu persiapan memasak.', NULL, '40000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870888124-492392036.jpg', '/uploads/items/item-1748870888124-492392036.jpg,/uploads/items/item-1748870888125-879822445.jpg'),
(33, 5, 1, 'Timbangan Dapur Digital Akurat 5kg', 'Timbangan dapur digital dengan kapasitas maksimal 5kg dan akurasi 1 gram. Dilengkapi layar LCD, fungsi tare (untuk menimbang dalam wadah), dan indikator baterai lemah. Penting untuk resep yang membutuhkan takaran presisi.', '60000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870913663-601686361.jpg', '/uploads/items/item-1748870913663-601686361.jpg,/uploads/items/item-1748870913663-631295783.jpg'),
(34, 5, 1, 'Cetakan Kue Silikon Bentuk Muffin (12 Rongga)', 'Cetakan kue berbahan silikon food-grade untuk membuat 12 buah muffin atau cupcake sekaligus. Fleksibel, anti lengket, dan mudah dilepaskan. Tahan suhu tinggi.', '35000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870951549-242402199.jpg', '/uploads/items/item-1748870951549-242402199.jpg,/uploads/items/item-1748870951550-125578016.jpg'),
(35, 5, 1, 'Blender Serbaguna Dengan Gelas Kaca', 'Blender bertenaga dengan gelas kaca kokoh berkapasitas 1.5 Liter. Ideal untuk membuat jus, smoothie, bumbu halus, hingga menghancurkan es. Mata pisau stainless steel tajam.', NULL, '50000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869352121-603239517.jpg', '/uploads/items/item-1748869352121-603239517.jpg'),
(36, 5, 1, 'Termometer Makanan Digital Probe', 'Termometer digital dengan probe stainless steel panjang untuk mengukur suhu internal daging, cairan, atau adonan. Layar LCD instan read. Penting untuk memastikan masakan matang sempurna dan aman.', '55000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869988700-885311951.jpg', '/uploads/items/item-1748869988700-885311951.jpg,/uploads/items/item-1748869988700-728537538.jpg,/uploads/items/item-1748869988701-241391587.jpg'),
(37, 5, 1, 'Buku Resep Masakan Nusantara Klasik', 'Kumpulan resep masakan tradisional Indonesia yang legendaris. Dilengkapi panduan langkah demi langkah yang mudah diikuti dan tips memasak. Buku wajib bagi pecinta kuliner nusantara.', '70000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870021975-699944997.jpg', '/uploads/items/item-1748870021975-699944997.jpg,/uploads/items/item-1748870021975-357978801.jpg,/uploads/items/item-1748870022006-535506056.jpg'),
(38, 5, 2, 'Kamera DSLR Canon EOS 2000D Kit 18-55mm', 'Kamera DSLR ideal untuk pemula dengan sensor 24.1MP. Dilengkapi lensa kit serbaguna 18-55mm IS II. Konektivitas Wi-Fi & NFC memudahkan transfer foto. Belajar fotografi dengan kualitas gambar profesional.', '3500000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870050049-769896161.jpg', '/uploads/items/item-1748870050049-769896161.jpg,/uploads/items/item-1748870050049-512726165.jpg'),
(39, 5, 2, 'Tripod Kamera Aluminium Ringan', 'Tripod tiga kaki berbahan aluminium yang ringan dan portabel. Ketinggian maksimal 160cm. Kepala tripod ball head memungkinkan pergerakan fleksibel. Cocok untuk stabilitas saat memotret atau merekam video.', NULL, '30000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870110254-509335967.jpg', '/uploads/items/item-1748870110254-509335967.jpg,/uploads/items/item-1748870110255-757011384.jpg'),
(40, 5, 2, 'Tas Kamera Selempang Dengan Pelindung Hujan', 'Tas kamera selempang yang ringkas namun muat 1 bodi kamera dan 2-3 lensa kecil/aksesoris. Padding tebal melindungi gear. Dilengkapi pelindung hujan terintegrasi. Nyaman untuk penggunaan sehari-hari atau traveling.', '120000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870135435-964705194.jpg', '/uploads/items/item-1748870135435-964705194.jpg,/uploads/items/item-1748870135435-968033761.jpg'),
(41, 5, 2, 'Filter Lensa UV 58mm Kualitas Optik', 'Filter UV bening untuk melindungi elemen depan lensa dari debu, goresan, dan sinar UV yang dapat mengurangi kejernihan gambar. Kualitas optik tinggi, tidak mengurangi ketajaman. Ukuran diameter 58mm.', NULL, '20000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870167721-314509787.jpg', '/uploads/items/item-1748870167721-314509787.jpg,/uploads/items/item-1748870167722-429510688.jpg'),
(42, 5, 2, 'Kit Lampu Studio Mini LED (2 Lampu)', 'Set 2 lampu studio mini LED dengan tripod dan diffuser. Memberikan pencahayaan tambahan yang lembut untuk foto produk, portrait kecil, atau video call. Ringkas dan mudah dipasang.', '75000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870199051-289802130.jpg', '/uploads/items/item-1748870199051-289802130.jpg,/uploads/items/item-1748870199052-734887434.jpg'),
(43, 5, 2, 'Reflektor Cahaya Lipat 5 in 1 (Diameter 60cm)', 'Reflektor fotografi multifungsi berdiameter 60cm, dapat dilipat ringkas. Terdiri dari 5 permukaan: Emas, Perak, Putih, Hitam, dan Transparan. Membantu mengontrol pencahayaan alami atau buatan.', '50000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870262999-408446624.jpg', '/uploads/items/item-1748870262999-408446624.jpg,/uploads/items/item-1748870263000-379582490.jpg'),
(44, 5, 2, 'Kartu Memori SDXC SanDisk Extreme Pro 128GB', 'Kartu memori SDXC berkecepatan tinggi (V30, U3) dengan kapasitas 128GB. Ideal untuk merekam video 4K dan foto burst. Transfer data cepat ke komputer. Wajib untuk fotografer/videografer.', NULL, '40000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870291372-824466092.jpg', '/uploads/items/item-1748870291372-824466092.jpg,/uploads/items/item-1748870291373-527637983.jpg'),
(45, 5, 2, 'Lensa Fix Portrait Nifty Fifty 50mm f/1.8', 'Lensa prime (fix) dengan focal length 50mm dan aperture lebar f/1.8. Menghasilkan bokeh (latar belakang blur) yang indah dan sangat baik di kondisi cahaya minim. Cocok untuk portrait.', '650000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870319739-616235827.jpg', '/uploads/items/item-1748870319739-616235827.jpg,/uploads/items/item-1748870319739-352462680.jpg'),
(46, 5, 3, 'E-reader Kindle Paperwhite 2024', 'E-reader terbaru dengan layar 6.8 inci bebas silau seperti kertas sungguhan. Cahaya latar yang hangat dan dapat disesuaikan. Tahan air. Baterai tahan hingga mingguan. Ribuan buku dalam genggaman.', '1700000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870379316-367554518.jpg', '/uploads/items/item-1748870379316-367554518.jpg,/uploads/items/item-1748870379317-364970542.jpg'),
(47, 5, 3, 'Lampu Baca Jepit LED (Rechargeable)', 'Lampu baca kecil dengan klip kuat untuk dijepitkan di buku atau e-reader. Menggunakan LED yang nyaman di mata, tingkat kecerahan dapat diatur. Baterai isi ulang via USB. Ideal untuk membaca di malam hari tanpa mengganggu orang lain.', NULL, '20000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870408854-592834542.jpg', '/uploads/items/item-1748870408854-592834542.jpg,/uploads/items/item-1748870408854-837052478.jpg'),
(48, 5, 3, 'Pembatas Buku Metal Motif Etnik', 'Pembatas buku unik terbuat dari metal dengan ukiran motif etnik yang indah. Kokoh dan tahan lama. Menjadi penanda halaman buku favorit Anda sekaligus aksesoris yang cantik.', '15000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870443678-318055280.jpg', '/uploads/items/item-1748870443678-318055280.jpg,/uploads/items/item-1748870443678-665134689.jpg'),
(49, 5, 3, 'Penyangga Buku Meja Kayu (Adjustable)', 'Penyangga buku portabel berbahan kayu berkualitas. Sudut kemiringan dapat diatur untuk posisi baca yang paling nyaman. Membantu mengurangi pegal pada leher dan tangan saat membaca buku tebal. Dilipat datar untuk penyimpanan.', NULL, '40000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870472542-812144542.jpg', '/uploads/items/item-1748870472542-812144542.jpg,/uploads/items/item-1748870472542-528480159.jpg'),
(50, 5, 3, 'Buku Fiksi Best-seller: \"Atomic Habits\"', 'Buku non-fiksi inspiratif karya James Clear tentang cara membangun kebiasaan baik dan menghilangkan kebiasaan buruk dengan metode perubahan kecil yang konsisten. Edisi terjemahan Bahasa Indonesia.', '95000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870502181-969825112.jpg', '/uploads/items/item-1748870502181-969825112.jpg,/uploads/items/item-1748870502181-943452663.jpg'),
(51, 5, 3, 'Buku Non-fiksi Populer: \"Sapiens: A Brief History of Humankind\"', 'Buku non-fiksi terkenal karya Yuval Noah Harari yang mengulas sejarah peradaban manusia dari sudut pandang baru. Edisi terjemahan Bahasa Indonesia. Wawasan mendalam tentang evolusi kita.', '120000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870529390-777797080.jpg', '/uploads/items/item-1748870529390-777797080.jpg,/uploads/items/item-1748870529390-598168217.jpg'),
(52, 5, 3, 'Sarung Pelindung E-reader Universal 6 Inci', 'Sarung pelindung universal yang kompatibel dengan sebagian besar e-reader berukuran 6 inci. Bahan kulit sintetis berkualitas melindungi dari goresan dan benturan ringan. Desain elegan.', '45000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870558235-908164944.jpg', '/uploads/items/item-1748870558235-908164944.jpg,/uploads/items/item-1748870558235-246633209.jpg'),
(53, 5, 3, 'Kaca Pembesar Dengan Lampu LED', 'Kaca pembesar genggam dengan pembesaran optik dan 2 lampu LED terintegrasi. Membantu membaca teks yang sangat kecil, peta, atau detail halus lainnya. Ringan dan mudah dibawa.', NULL, '25000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870587694-70919606.jpg', '/uploads/items/item-1748870587694-70919606.jpg,/uploads/items/item-1748870587694-584530797.jpg'),
(54, 5, 3, 'Meja Lipat Portabel Untuk Laptop & Membaca', 'Meja lipat ringkas yang dapat digunakan di atas kasur atau sofa. Kaki dapat dilipat, dilengkapi slot untuk tablet/ponsel dan cup holder. Permukaan luas untuk menopang buku atau laptop saat membaca/bekerja.', NULL, '70000.00', 0, 1, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870625101-991759715.jpg', '/uploads/items/item-1748870625101-991759715.jpg,/uploads/items/item-1748870625102-67686853.jpg'),
(55, 5, 3, 'Set Pulpen Gel Warna-warni (12 Warna)', 'Set 12 pulpen gel dengan tinta berbagai warna cerah. Ujung pena halus 0.5mm. Ideal untuk membuat catatan, menandai bagian penting di buku (jika diizinkan), atau membuat jurnal bacaan.', '20000.00', NULL, 1, 0, '0.00', 'available', '2025-06-01 22:47:18', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748870674083-124650353.jpg', '/uploads/items/item-1748870674083-124650353.jpg,/uploads/items/item-1748870674084-873593785.jpg,/uploads/items/item-1748870674087-856845978.jpg'),
(56, 5, 1, 'Set Pisau Dapur Stainless Steel 5-in-1', 'Paket lengkap pisau dapur anti karat dengan pegangan ergonomis, ideal untuk kebutuhan memasak harian.', '85000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869098536-396252396.jpg', '/uploads/items/item-1748869098536-396252396.jpg'),
(57, 5, 1, 'Panci Anti Lengket Teflon 24cm', 'Panci masak dengan lapisan anti lengket berkualitas tinggi, cocok untuk menumis dan memasak dengan sedikit minyak.', '90000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869315520-377753291.png', '/uploads/items/item-1748869315520-377753291.png'),
(58, 5, 1, 'Kompor Induksi Digital 2000W', 'Kompor hemat energi dengan kontrol suhu digital dan fitur pengaman otomatis, praktis dan efisien.', NULL, '70000.00', 0, 1, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869327166-178616130.jpg', '/uploads/items/item-1748869327166-178616130.jpg'),
(59, 5, 1, 'Blender Serbaguna Philips 600W', 'Blender dengan 3 kecepatan, cocok untuk membuat jus, smoothie, dan menghaluskan bumbu dapur.', '210000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869520836-185181686.jpg', '/uploads/items/item-1748869520836-185181686.jpg'),
(60, 5, 1, 'Loyang Kue Persegi Anti Lengket', 'Loyang berkualitas food-grade untuk memanggang kue, roti, atau lasagna dengan hasil matang merata.', '45000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869381190-702673080.jpg', '/uploads/items/item-1748869381190-702673080.jpg'),
(61, 5, 1, 'Panci Listrik Serbaguna 1.5L', 'Panci listrik multifungsi dengan kapasitas 1.5 liter, cocok untuk memasak mie, merebus telur, hingga menghangatkan sup. Dilengkapi dengan pengatur suhu dan permukaan anti lengket.', NULL, '50000.00', 0, 1, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869399657-999033074.png', '/uploads/items/item-1748869399657-999033074.png'),
(62, 5, 1, 'Teko Elektrik Stainless Steel 1.8L', 'Teko pemanas air otomatis dengan daya 1500W dan kapasitas besar, terbuat dari bahan stainless steel anti karat. Fitur mati otomatis saat air mendidih dan sistem perlindungan dari kekeringan.', '120000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869418405-166842747.jpg', '/uploads/items/item-1748869418405-166842747.jpg'),
(63, 5, 1, 'Toples Kaca Kedap Udara 1000ml', 'Toples penyimpanan makanan berbahan kaca tebal dengan tutup kedap udara berbahan bambu. Ideal untuk menyimpan kopi, biskuit, atau bahan dapur kering agar tetap segar lebih lama.', '30000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869437527-679926691.jpg', '/uploads/items/item-1748869437527-679926691.jpg'),
(64, 5, 2, 'Kamera Mirrorless Canon EOS M50 Mark II', 'Kamera mirrorless ringan dengan kemampuan video 4K dan layar sentuh putar, ideal untuk vlogging dan fotografi sehari-hari.', NULL, '90000.00', 0, 1, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869455156-796176828.jpg', '/uploads/items/item-1748869455156-796176828.jpg'),
(65, 5, 2, 'Lensa Fix Sony FE 50mm f/1.8', 'Lensa prime dengan bukaan besar untuk efek bokeh yang memukau, cocok untuk potret dan kondisi cahaya rendah.', '1350000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869469760-673797463.jpg', '/uploads/items/item-1748869469760-673797463.jpg'),
(66, 5, 2, 'Tripod Aluminium Manfrotto Compact', 'Tripod ringan dan kokoh dengan kepala ball head, mendukung kamera DSLR dan mirrorless untuk pengambilan gambar stabil.', '280000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869492029-73193247.jpg', '/uploads/items/item-1748869492029-73193247.jpg'),
(67, 5, 2, 'Lighting Kit LED Neewer 660', 'Paket lampu LED serbaguna dengan intensitas dan suhu warna yang dapat diatur, sempurna untuk studio kecil.', NULL, '60000.00', 0, 1, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869133361-845257597.png', '/uploads/items/item-1748869133361-845257597.png'),
(68, 5, 2, 'Tas Kamera Lowepro ProTactic BP 450 AW II', 'Tas kamera tahan air dengan kompartemen fleksibel untuk kamera, lensa, drone, dan aksesori lainnya.', '800000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869849700-538534614.jpg', '/uploads/items/item-1748869849700-538534614.jpg'),
(69, 5, 3, '“Atomic Habits” - James Clear', 'Panduan praktis membangun kebiasaan kecil yang membawa perubahan besar dalam hidup.', '95000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869725230-190331346.jpg', '/uploads/items/item-1748869725230-190331346.jpg'),
(70, 5, 3, '“Laut Bercerita” - Leila S. Chudori', 'Novel fiksi sejarah yang menyentuh, menceritakan kisah aktivis era Orde Baru yang hilang secara misterius.', '87000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869746996-230879086.jpg', '/uploads/items/item-1748869746996-230879086.jpg'),
(71, 5, 3, '“Sapiens: A Brief History of Humankind” - Yuval Noah Harari', 'Buku non-fiksi populer yang membahas evolusi manusia dari masa purba hingga era modern.', '120000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869773467-866999607.jpg', '/uploads/items/item-1748869773467-866999607.jpg'),
(72, 5, 3, '“Filosofi Teras” - Henry Manampiring', 'Pengantar pemikiran stoikisme yang relevan untuk mengelola emosi dan kebahagiaan dalam hidup sehari-hari.', '88000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869799063-810559292.jpg', '/uploads/items/item-1748869799063-810559292.jpg'),
(73, 5, 3, '“Rich Dad Poor Dad” - Robert T. Kiyosaki', 'Buku edukatif tentang literasi finansial dan perbedaan mindset antara orang kaya dan miskin.', '99000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:30:27', '12', 'SUMATERA UTARA', '1275', 'KOTA MEDAN', '/uploads/items/item-1748869819530-508384295.jpg', '/uploads/items/item-1748869819530-508384295.jpg'),
(74, 6, 1, 'Talenan Kayu Pinus', 'Talenan kayu ini terbuat dari kayu pinus pilihan, proses pengeringannya menggunakan panas alami dari sinar Matahari, sehingga kayu lebih kering dan tidak gampang patah maupun melengkung. penggunaan kaki membuat lebih kokoh dan tidak mudah bergeser saat digunakan mengiris.', '45000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928298698-657155808.jpg', '/uploads/items/item-1748928298698-657155808.jpg'),
(75, 6, 1, 'Mixer Kecepatan Tinggi', 'Mixer tangan dengan pegangan yang terbuat dari plastik kuat guna memudahkan dalam memegang mixer. Mixer ini menawarkan fleksibilitas dengan 7 tingkat kecepatan yang dapat disesuaikan dengan kebutuhan sehingga dapat menghasilkan adonan kue yang mengembang dan merata dengan lebih cepat.', NULL, '50000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928313406-714829221.jpg', '/uploads/items/item-1748928313406-714829221.jpg'),
(76, 6, 1, 'Pisau Daging Tajam Carbon', 'Dibuat dari baja karbon yang ditempa sehingga lebih tajam, kokoh, dan tahan lama. Handlenya dibuat ergonomis sehingga memotong daging lebih nyaman dan presisi. Pisau yang satu ini bisa Anda andalkan untuk memotong daging dan tulang.', '80000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928327351-595601621.jpg', '/uploads/items/item-1748928327351-595601621.jpg'),
(77, 6, 1, 'Wajan Mini Anti Lengket', 'Hadir dengan material aluminium berlapis non stick sehingga masakan tak mudah lengket. Wajan ini memudahkan Anda menggoreng bahan makanan kecil seperti telur omelet ataupun sekedar menggoreng patty burger.', '45000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928340860-773949612.jpg', '/uploads/items/item-1748928340860-773949612.jpg'),
(78, 6, 1, 'Spatula Anti Lengket', 'Bilahnya yang tipis memudahkan saat menyelipkan spatula di bawah makanan, sedangkan ukurannya yang besar memungkinkan Anda membalikkan makanan dengan mudah.', NULL, '25000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928360375-573238807.png', '/uploads/items/item-1748928360375-573238807.png'),
(79, 6, 2, 'Lensa Sony FE 50mm f/1.2 GM', 'Merupakan Lensa prime classic normal focal length dengan desain yang diperbarui dan ramping, lensa dicirikan oleh kecepatan dan form factor yang relatif ringkas. Selain itu, sebagai lensa G Master, lensa ini memiliki desain optik canggih untuk citra yang terkoreksi dengan baik dengan tingkat ketajaman dan kejernihan yang tinggi.', '2250000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928374577-687421656.jpg', '/uploads/items/item-1748928374577-687421656.jpg'),
(80, 6, 2, 'Lensa Sony FE 70-200mm f/2.8 GM OSS', 'Sony FE 70-200mm f/2.8 GM OSS merupakan lensa telefoto zoom untuk profesional, sebagai anggota dari lensa seri G Master, lensa ini dapat memberikan ketajaman tinggi bersama bokeh yang halus dan bersih dan dengan penyimpangan minimum absolut.', '2500000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928389001-618754463.jpg', '/uploads/items/item-1748928389001-618754463.jpg'),
(81, 6, 2, 'Lensa Sony FE 24-70mm f/2.8 GM II', 'Sony FE 24-70mm f/2.8 GM II tidak hanya lebih kecil dan lebih ringan dari generasi sebelumnya, tetapi juga dilengkapi berbagai peningkatan optik, pemfokusan, dan penanganan, yang melayani foto dan aplikasi video.', NULL, '120000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928403329-402118843.jpg', '/uploads/items/item-1748928403329-402118843.jpg'),
(82, 6, 2, 'Lensa Sony FE 16-35mm f/2.8 GM', 'Sony FE 16-35mm f/2.8 GM merupakan lensa wide-angle zoom yang cepat dan fleksibel. Lensa ini memiliki aperture f / 2.8 konstan dan juga menawarkan kinerja yang konsisten sepanjang rentang zoom dan manfaat lainnya ketika bekerja dalam kondisi low-light.', '1950000.00', NULL, 1, 0, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928415895-565582920.jpg', '/uploads/items/item-1748928415895-565582920.jpg'),
(83, 6, 2, 'Lensa Sony FE 200-600mm f/5.6 – 6.3 G OSS', 'Sony FE 200-600mm F5.6-6.3 G OSS mencakup rentang telefoto serbaguna, lensa zoom fleksibel yang sangat cocok untuk aplikasi alam, margasatwa, dan olahraga.', NULL, '150000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928428064-976181991.jpg', '/uploads/items/item-1748928428064-976181991.jpg'),
(84, 6, 3, '“Kambing Jantan” – Raditya Dika', 'Novel komedi ini bercerita tentang keseharian Raditya Dika semasa berkuliah di Australia.', NULL, '20000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928443095-748061399.jpg', '/uploads/items/item-1748928443095-748061399.jpg'),
(85, 6, 3, '“Seporsi Mie Ayam Sebelum Mati” – Brian Khrisna', 'Novel ini mengangkat isu kesehatan mental, khususnya depresi, dengan sudut pandang yang unik.', NULL, '25000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928464781-745411631.png', '/uploads/items/item-1748928464781-745411631.png'),
(86, 6, 3, '“The Psychology of Money” – Morgan Housel', 'Merupakan buku yang mengeksplorasi hubungan antara psikologi dan keuangan. Buku ini tidak hanya membahas teori keuangan, tetapi juga bagaimana emosi, perilaku, dan pengalaman pribadi kita memengaruhi cara kita berinteraksi dengan uang.', NULL, '27000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928478452-823062042.jpg', '/uploads/items/item-1748928478452-823062042.jpg'),
(87, 6, 3, '“Madilog: Materialisme, Dialektika, dan Logika” – Tan Malaka', 'Karya yang memberikan dasar pemahaman tentang materialisme, dialektika, dan logika bagi para kader dan pemimpin pergerakan revolusioner di Indonesia.', NULL, '30000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928491598-478132020.jpg', '/uploads/items/item-1748928491598-478132020.jpg'),
(88, 6, 3, '“Namaku Alam” – Leila S. Chudori', 'Buku ini mengisahkan perjalanan hidup Segara Alam, seorang remaja yang mengalami kesulitan mencari identitas diri akibat stigma \"keluarga tapol\" pasca peristiwa 1965.', NULL, '23000.00', 0, 1, '0.00', 'available', '2025-06-02 06:34:57', '51', 'BALI', '5101', 'KABUPATEN JEMBRANA', '/uploads/items/item-1748928505113-582592278.png', '/uploads/items/item-1748928505113-582592278.png');

-- --------------------------------------------------------

--
-- Table structure for table `messages`
--

CREATE TABLE `messages` (
  `id` int NOT NULL,
  `sender_id` int NOT NULL,
  `receiver_id` int NOT NULL,
  `item_id` int DEFAULT NULL,
  `transaction_id` int DEFAULT NULL,
  `content` text NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `messages_community`
--

CREATE TABLE `messages_community` (
  `id` int NOT NULL,
  `sender_id` int NOT NULL,
  `province_id` int NOT NULL,
  `content` text NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `messages_community`
--

INSERT INTO `messages_community` (`id`, `sender_id`, `province_id`, `content`, `created_at`) VALUES
(1, 5, 12, 'halo', '2025-05-30 10:27:27'),
(2, 5, 12, 'abangku', '2025-05-30 10:41:34'),
(3, 4, 12, 'yah fiq', '2025-05-30 10:42:32'),
(4, 5, 12, 'apa cs', '2025-05-30 10:42:43'),
(5, 6, 51, 'horas', '2025-05-30 11:32:06'),
(6, 4, 12, 'halo ges', '2025-05-31 04:49:26'),
(7, 4, 12, 'yud', '2025-05-31 05:03:55'),
(8, 4, 12, 'oi', '2025-05-31 08:45:02'),
(9, 4, 12, 'jajan', '2025-05-31 08:45:52'),
(10, 4, 12, 'halo', '2025-05-31 08:48:48'),
(11, 4, 12, 'halo', '2025-05-31 08:49:30'),
(12, 4, 12, 'halo ges', '2025-05-31 08:54:58'),
(13, 4, 12, 'ges', '2025-05-31 08:58:07'),
(14, 4, 12, 'pada ngapain', '2025-05-31 08:59:29'),
(15, 4, 12, 'ajsfdsfsfdfdf', '2025-05-31 09:00:03'),
(16, 4, 12, 'abangku', '2025-05-31 09:02:31'),
(17, 4, 12, 'lagi ngapain', '2025-05-31 09:02:34'),
(18, 4, 12, 'keren keren', '2025-05-31 09:02:37'),
(19, 4, 12, 'ih lagi ngapain kelen abangda', '2025-05-31 09:02:59'),
(20, 4, 12, 'gaada men maen aja, orang abang?', '2025-05-31 09:03:16'),
(21, 4, 12, 'gabut nich', '2025-05-31 09:03:25'),
(22, 4, 12, 'ges', '2025-05-31 09:03:44'),
(23, 4, 12, 'bug kah?', '2025-05-31 09:03:52'),
(24, 4, 12, 'ada bug kah?', '2025-05-31 09:03:59'),
(25, 4, 12, 'haloo', '2025-05-31 09:04:06'),
(26, 4, 12, 'hajan', '2025-05-31 09:04:14'),
(27, 4, 12, 'halo kak', '2025-05-31 09:12:21'),
(28, 4, 12, 'halo om', '2025-05-31 09:15:50'),
(29, 5, 12, 'sore all', '2025-05-31 09:19:23'),
(30, 4, 12, 'p', '2025-05-31 12:10:13'),
(31, 4, 12, 'malam all', '2025-05-31 12:10:16'),
(32, 5, 12, 'malam', '2025-05-31 12:10:23'),
(33, 4, 12, 'halo ges', '2025-05-31 16:20:29'),
(34, 4, 12, 'lagi pada ngapain?', '2025-05-31 16:20:37'),
(35, 4, 12, 'pagi all', '2025-05-31 17:15:00'),
(36, 5, 12, 'pagii', '2025-05-31 17:15:06'),
(37, 5, 12, 'lagi ngoding kah?', '2025-05-31 17:16:43'),
(38, 4, 12, 'iya bang lagi ngoding hehehe', '2025-05-31 17:16:57'),
(39, 4, 12, 'abang lagi ngoding juga kah?', '2025-05-31 17:17:04'),
(40, 5, 12, 'iya bang, lagi ngoding juga', '2025-05-31 17:17:12'),
(41, 5, 12, 'banyak bet kerjaan ini wkwkw', '2025-05-31 17:17:46'),
(42, 7, 12, 'abangda', '2025-06-01 07:39:07'),
(43, 8, 12, 'halow', '2025-06-01 08:36:53'),
(44, 7, 12, 'eyo', '2025-06-01 08:37:02'),
(45, 8, 12, 'abangda pagi smeunaya', '2025-06-01 22:18:44'),
(46, 6, 51, 'hai', '2025-06-03 05:29:01');

-- --------------------------------------------------------

--
-- Table structure for table `notifications`
--

CREATE TABLE `notifications` (
  `id` int NOT NULL,
  `user_id` int NOT NULL,
  `message` text NOT NULL,
  `is_read` tinyint(1) DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `type` enum('transaction','message','rent_reminder') NOT NULL DEFAULT 'transaction',
  `transaction_id` int DEFAULT NULL,
  `related_user_id` int DEFAULT NULL,
  `scheduled_date` date DEFAULT NULL,
  `reminder_day` tinyint DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `reviews`
--

CREATE TABLE `reviews` (
  `id` int NOT NULL,
  `user_id` int NOT NULL,
  `item_id` int NOT NULL,
  `comment` text,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `transactions`
--

CREATE TABLE `transactions` (
  `id` int NOT NULL,
  `buyer_id` int NOT NULL,
  `item_id` int NOT NULL,
  `type` enum('rent','buy') NOT NULL,
  `status` enum('pending','ongoing','returned','late','cancelled','completed') NOT NULL DEFAULT 'pending',
  `payment_method` enum('cod') DEFAULT 'cod',
  `total_price` decimal(10,2) DEFAULT NULL,
  `rent_start_date` date DEFAULT NULL,
  `rent_end_date` date DEFAULT NULL,
  `deposit_paid` decimal(10,2) DEFAULT '0.00',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `photo` varchar(255) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `province_id` int DEFAULT NULL,
  `province_name` varchar(100) DEFAULT NULL,
  `city_id` int DEFAULT NULL,
  `city_name` varchar(100) DEFAULT NULL,
  `push_subscription` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `name`, `email`, `password`, `photo`, `created_at`, `province_id`, `province_name`, `city_id`, `city_name`, `push_subscription`) VALUES
(4, 'Yudi', 'yudi@gmail.com', '$2b$12$mbJ60aAs.XW/Q0QkkzCgmuG3jXOWrXGagJSYB.4ueaaK0DrgfDRVG', NULL, '2025-05-29 01:42:55', 12, 'SUMATERA UTARA', 1275, 'KOTA MEDAN', '{\"endpoint\":\"https://wns2-pn1p.notify.windows.com/w/?token=BQYAAAAaBVsBX%2b5Gf%2bzutg1ZUZCmj%2bqp1X70raWammSc2e6j3TzSvlvK%2bcs5CjHA6%2bUVURptHEjBs6lyCB7MqonTJcaSV%2f72fWMre54KZ39L52v0uS%2bh%2bnUmj4VdXdhjVX6pwPzBBVs1i72Ilbcl1xrkV1K1FUPgFb1RLkmjedWYSUmb6qCoTBc%2b0%2bP6m3FZg9CDPRxSE9Z9X%2bWRj6M5NAitTHR7ptxKsFmp5p0VKzh9tWxqAv4bTjFtK6N5BZiXMN7m9Qgn5TZXmeAaLMB%2b7zIKe9xI%2bdlwA7zmbeDY%2bXEGoImckFd00Low9yYC2Bf92wHNwGc%3d\",\"keys\":{\"p256dh\":\"BIK74JYxBRTRKRsxvTZvmqhRun7h0UX757BPJMNd7H138GlxxYIZolB2tMMdqwkE4YqDer245kPfTrc9d0SlgSo=\",\"auth\":\"P/R7F3Y4iPA37aMNciCl1A==\"}}'),
(5, 'fiqri', 'fiqri@gmail.com', '$2b$12$RdQParH6ftkId3aQUbFPIeyGcwUzp26uYfdNHStoTaqEmNxRQMRd6', NULL, '2025-05-29 01:46:49', 12, 'SUMATERA UTARA', 1275, 'KOTA MEDAN', '{\"endpoint\":\"https://wns2-pn1p.notify.windows.com/w/?token=BQYAAAB41wlseBN1%2fwDsePQEgTL3xmz7eM0rcO7DDktj6dulsePb%2fvkODASozZlZtqlk0BKEsOukGgboDqjnS5fv71lvbg5Uh%2b3uTxfE1Kj2pQ02nL9VDof7tX8Pd%2b7TJR3k3llLOOzx7mvmPF7qfcmBVEnwsOTPwlJazNhnU6fIrEzD%2buc6zHGuF8ag18P6l7LYcE2lUyrua0KNISm1eKQU6nqgSVLb2VuF8zRJXd6NF0gzQFoFyJFSsw9L3xAq3rsU6B06Pgt9EKZPpOkt%2bYGMlp9s3NUOvVVONDoaL8TOnCeOq%2bYT6WDOSy6vYTaJYNc9hux9J1H5mUYDo1HCsI8kM%2bp4\",\"keys\":{\"p256dh\":\"BOPdv9ejgpP66kZfxLZB7ccV2aEw1KUITSXVbeNWUyi0ZhyZ2GiUgQoCb6DjK8n4NslAi/QCoPXt8E/1lghtUQ8=\",\"auth\":\"c6DIXP2iSYFzvMpBo2F+xA==\"}}'),
(6, 'yoga', 'yoga@gmail.com', '$2b$12$I9dZSiqriq5/MlbhabkiZeB5ha1gZlAE2/79NwaMxbjCsUyZryC8u', NULL, '2025-05-30 10:46:10', 51, 'BALI', 5101, 'KABUPATEN JEMBRANA', '{\"endpoint\":\"https://fcm.googleapis.com/fcm/send/doc1QMSIEVI:APA91bG7Omw0lf1rJQ5hW1gOvUhy7bnSJkjmOolROqG9ctzaTzl_OByFVbmW710mkD93d2ACkgwyCvUlq4ZwG_DhoGEzU56KYk15SST0ovbpZaZAnR0xTViL8m9aAyj13-5_YHX7AYNm\",\"keys\":{\"p256dh\":\"BG/JAILCThIQtUfkeNo52eQ36z51ec3x04edpwDMbJO2aFpvAlld6yaS3CFPnJZAvJdk4+YqaMqOe/O5SfMhvI4=\",\"auth\":\"Hgo+SYj5Owt8LLdbkSbpjA==\"}}'),
(7, 'yudii', 'yudii@gmail.com', '$2b$12$EIA4/lZw7cxjBIjoLQJlgOSXQbX89MI2d6fAWN.l2B50zefS.Zgum', NULL, '2025-06-01 06:10:35', 12, 'SUMATERA UTARA', 1276, 'KOTA BINJAI', '{\"endpoint\":\"https://fcm.googleapis.com/fcm/send/e1CGHgF0TaI:APA91bEPNFCYj4GXMD_y-6WFMDzVtcEQNM5GRofsE2y1B8wwfLWwrnzCBHBWzMo7bm7IUiQEqVzsOYeCrhiTmsE4OqexqHI-wwhqGYx3ojlA8Z4J_BKsdl5CA7o3TVgBKAFm6thCITcc\",\"keys\":{\"p256dh\":\"BH1LMUqLb6rgKrR2eccU/wRqk3lpdVruh/s0KOOp6iE7OGFIIb6fQA9EjcBA5ha7cTpgWBx6tOpSjBFgPR8aGYI=\",\"auth\":\"PnzcanmRhwbgC6Byzsxtdg==\"}}'),
(8, 'yogi', 'yogi@gmail.com', '$2b$12$4NrS9mUmYQAsQgk/VxvPFumwNVAszmMbEWcmpsQt2N8yMY9Omc79O', NULL, '2025-06-01 08:35:48', 12, 'SUMATERA UTARA', 1277, 'KOTA PADANGSIDIMPUAN', '{\"endpoint\":\"https://wns2-pn1p.notify.windows.com/w/?token=BQYAAAD2cg5kayFXMUnM5CpAZyoUhbCv07c7606FXV%2bhCOqOFrgENq5KQ9T2UQ9bkqvRPDA%2bqvXCEbgvJsvWNNBsjrnTROqfjMVx8cwGnqPhulCwlGOvJpPMbXRB%2fMRH%2b41qchRSKVUlm0hVJBxogeqdSZGgehuCsiBeHwQ6JOnNk95WxmopMfTLUeg09FZQQUETHcpuldXS8buM7XpxFPfMdR4cAwHBk9A0eRe%2fLv%2b8oUduR1A%2b69U5IvOrE0AWFEtzlzBDAxdTjD6dPhV2nUkHj5UgEfPH5OM07SJEOoC67%2bYI5KrCJvaz3%2bqUiTKi7T1piQo%3d\",\"keys\":{\"p256dh\":\"BK8Oi/dy2s5XbCeiBGT5cr9ZWTnSJudhs2vh83mllL3fan+DMwpjazQw85f47g0bGvIBQMeFwZL0rMh7NbW+fj4=\",\"auth\":\"7zp7EeQHxK41HYFDqZMZ1w==\"}}');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `items`
--
ALTER TABLE `items`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `category_id` (`category_id`);

--
-- Indexes for table `messages`
--
ALTER TABLE `messages`
  ADD PRIMARY KEY (`id`),
  ADD KEY `sender_id` (`sender_id`),
  ADD KEY `receiver_id` (`receiver_id`),
  ADD KEY `item_id` (`item_id`),
  ADD KEY `transaction_id` (`transaction_id`);

--
-- Indexes for table `messages_community`
--
ALTER TABLE `messages_community`
  ADD PRIMARY KEY (`id`),
  ADD KEY `sender_id` (`sender_id`);

--
-- Indexes for table `notifications`
--
ALTER TABLE `notifications`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_user_type` (`user_id`,`type`),
  ADD KEY `idx_scheduled_date` (`scheduled_date`),
  ADD KEY `idx_transaction` (`transaction_id`);

--
-- Indexes for table `reviews`
--
ALTER TABLE `reviews`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `item_id` (`item_id`);

--
-- Indexes for table `transactions`
--
ALTER TABLE `transactions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `buyer_id` (`buyer_id`),
  ADD KEY `item_id` (`item_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `categories`
--
ALTER TABLE `categories`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `items`
--
ALTER TABLE `items`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=89;

--
-- AUTO_INCREMENT for table `messages`
--
ALTER TABLE `messages`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=32;

--
-- AUTO_INCREMENT for table `messages_community`
--
ALTER TABLE `messages_community`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=47;

--
-- AUTO_INCREMENT for table `notifications`
--
ALTER TABLE `notifications`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=47;

--
-- AUTO_INCREMENT for table `reviews`
--
ALTER TABLE `reviews`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT for table `transactions`
--
ALTER TABLE `transactions`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=23;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `items`
--
ALTER TABLE `items`
  ADD CONSTRAINT `items_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `items_ibfk_2` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`);

--
-- Constraints for table `messages`
--
ALTER TABLE `messages`
  ADD CONSTRAINT `messages_ibfk_1` FOREIGN KEY (`sender_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `messages_ibfk_2` FOREIGN KEY (`receiver_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `messages_ibfk_3` FOREIGN KEY (`item_id`) REFERENCES `items` (`id`),
  ADD CONSTRAINT `messages_ibfk_4` FOREIGN KEY (`transaction_id`) REFERENCES `transactions` (`id`);

--
-- Constraints for table `messages_community`
--
ALTER TABLE `messages_community`
  ADD CONSTRAINT `messages_community_ibfk_1` FOREIGN KEY (`sender_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `notifications`
--
ALTER TABLE `notifications`
  ADD CONSTRAINT `notifications_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);

--
-- Constraints for table `reviews`
--
ALTER TABLE `reviews`
  ADD CONSTRAINT `reviews_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `reviews_ibfk_2` FOREIGN KEY (`item_id`) REFERENCES `items` (`id`);

--
-- Constraints for table `transactions`
--
ALTER TABLE `transactions`
  ADD CONSTRAINT `transactions_ibfk_1` FOREIGN KEY (`buyer_id`) REFERENCES `users` (`id`),
  ADD CONSTRAINT `transactions_ibfk_2` FOREIGN KEY (`item_id`) REFERENCES `items` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
