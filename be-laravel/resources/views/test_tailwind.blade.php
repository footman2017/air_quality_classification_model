<!doctype html>
<html>

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
    {{-- <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script> --}}
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>

<body id="landing-page-testing">
    <div class="container mx-auto p-4 w-9/12">
        <div class="grid grid-cols-2 mb-4">
            <div class="flex">
                <p class="text-xl font-semibold mr-4">Lokasi</p>
                <select id="select_location">
                    <option value="BANDUNG">
                        Bandung Cihapit
                    </option>
                    <option value="DKI1">
                        DKI Bundaran HI
                    </option>
                    <option value="DKI2">
                        DKI Kelapa Gading
                    </option>
                    <option value="DKI3">
                        DKI Jagakarsa
                    </option>
                </select>
            </div>
            <div>

            </div>
        </div>
        <div class="grid grid-cols-6 gap-4">
            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">PM<sub>10</sub></p>
                <p id="pm10" class="text-xl mt-2"></p>
            </div>

            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">PM<sub>25</sub></p>
                <p id="pm25" class="text-xl mt-2"></p>
            </div>

            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">SO<sub>2</sub></p>
                <p id="so2" class="text-xl mt-2"></p>
            </div>

            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">CO</p>
                <p id="co" class="text-xl mt-2"></p>
            </div>

            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">O<sub>3</sub></p>
                <p id="o3" class="text-xl mt-2"></p>
            </div>

            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">NO<sub>2</sub></p>
                <p id="no2" class="text-xl mt-2"></p>
            </div>
        </div>

        <div class="grid grid-cols-2 gap-4 mt-4">
            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">Hasil Prediksi</sub></p>
                <p id="prediction" class="text-xl mt-2"></p>
            </div>

            <div class="text-center border border-solid rounded-md border-slate-400">
                <p class="text-2xl font-semibold">Akurasi Prediksi</p>
                <p id="akurasi" class="text-xl mt-2"></p>
            </div>
        </div>

        <div class="grid grid-cols-2 gap-4 mt-4">
            <div class="chart-container" style="position: relative; height:min-content; width:100%">
                <canvas id="pm10Chart"></canvas>
            </div>
            <div class="chart-container" style="position: relative; height:min-content; width:100%">
                <canvas id="pm25Chart"></canvas>
            </div>
            <div class="chart-container" style="position: relative; height:min-content; width:100%">
                <canvas id="so2Chart"></canvas>
            </div>
            <div class="chart-container" style="position: relative; height:min-content; width:100%">
                <canvas id="coChart"></canvas>
            </div>
            <div class="chart-container" style="position: relative; height:min-content; width:100%">
                <canvas id="o3Chart"></canvas>
            </div>
            <div class="chart-container" style="position: relative; height:min-content; width:100%">
                <canvas id="no2Chart"></canvas>
            </div>
        </div>
    </div>
</body>

</html>
