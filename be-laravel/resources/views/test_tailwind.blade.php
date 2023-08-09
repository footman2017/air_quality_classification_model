<!doctype html>
<html>

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>

<body id="landing-page">
    <div class="container mx-auto p-4">
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
                <p class="text-2xl font-semibold">Akurasi</p>
                <p id="akurasi" class="text-xl mt-2"></p>
            </div>
        </div>
    </div>
</body>

</html>
