import Swal from "sweetalert2";
import Chart from "chart.js/auto";

$(function () {
    let $root = $("#landing-page-testing");
    if (!$root.length) return;

    const pm10Elm = $("#pm10");
    const pm2Elm = $("#pm25");
    const so2Elm = $("#so2");
    const coElm = $("#co");
    const o3Elm = $("#o3");
    const no2Elm = $("#no2");
    const hasilPrediksiElm = $("#prediction");
    const AkurasiElm = $("#akurasi");
    const selectLocation = $("#select_location");

    let chartPm10, chartPm25, chartSo2, chartCo, chartO3, chartNo2;

    const categori = ["BAIK", "SEDANG", "TIDAK SEHAT"];

    function showLoading() {
        Swal.fire({
            allowOutsideClick: false,
            allowEscapeKey: false,
            width: "100px",
            background: "transparent",
            didOpen: () => {
                Swal.showLoading();
            },
        });
    }

    function hitBe(locationId = "BANDUNG") {
        pm10Elm.html("");
        pm2Elm.html("");
        so2Elm.html("");
        coElm.html("");
        o3Elm.html("");
        no2Elm.html("");
        hasilPrediksiElm.html("");
        AkurasiElm.html("");

        let config = {
            method: "get",
            maxBodyLength: Infinity,
            url: `${window.origin}/api/get-latest-prediction/${locationId}`,
            headers: {},
        };

        showLoading();

        axios
            .request(config)
            .then((response) => {
                Swal.close();
                if (response.data.status && response.data.data.length != 0) {
                    pm10Elm.html(response.data.data.pm10);
                    pm2Elm.html(response.data.data.pm25);
                    so2Elm.html(response.data.data.so2);
                    coElm.html(response.data.data.co);
                    o3Elm.html(response.data.data.o3);
                    no2Elm.html(response.data.data.no2);
                    hasilPrediksiElm.html(
                        categori[response.data.data.prediction_result]
                    );
                    AkurasiElm.html(`${response.data.data.accuracy} %`);

                    const inputDate = new Date(response.data.data.created_at);

                    const options = {
                        year: "numeric",
                        month: "long",
                        day: "numeric",
                    };
                    const formattedDate = inputDate.toLocaleDateString(
                        "id-ID",
                        options
                    );

                    console.log(formattedDate); // Output: "9 Agustus 2023"

                    get24LatestData();
                }
            })
            .catch((error) => {
                console.log(error);
            });
    }

    const formatWaktu = (timestamp) => {
        const date = new Date(timestamp);
        const day = date.getDate().toString().padStart(2, "0");
        const month = (date.getMonth() + 1).toString().padStart(2, "0");
        const hours = date.getHours().toString().padStart(2, "0");
        const minutes = date.getMinutes().toString().padStart(2, "0");
        return `${day}-${month} ${hours}:${minutes}`;
    };

    const createChart = (idElement, labelTitle, data) => {
        return new Chart(document.getElementById(idElement), {
            type: "line",
            data: {
                labels: data.map((row) => row.waktu),
                datasets: [
                    {
                        label: labelTitle,
                        data: data.map((row) => row.count),
                    },
                ],
            },
        });
    };

    function get24LatestData() {
        let config = {
            method: "get",
            maxBodyLength: Infinity,
            url: `${
                window.origin
            }/api/get-24-prediction/${selectLocation.val()}`,
            headers: {},
        };

        showLoading();

        axios
            .request(config)
            .then((response) => {
                Swal.close();
                if (response.data.status && response.data.data.length != 0) {
                    console.log(response.data.data);

                    const dataPm10 = response.data.data.map((item) => ({
                        waktu: formatWaktu(item.created_at),
                        count: item.pm10,
                    }));
                    const dataPm25 = response.data.data.map((item) => ({
                        waktu: formatWaktu(item.created_at),
                        count: item.pm25,
                    }));
                    const dataSo2 = response.data.data.map((item) => ({
                        waktu: formatWaktu(item.created_at),
                        count: item.so2,
                    }));
                    const dataCo = response.data.data.map((item) => ({
                        waktu: formatWaktu(item.created_at),
                        count: item.co,
                    }));
                    const dataO3 = response.data.data.map((item) => ({
                        waktu: formatWaktu(item.created_at),
                        count: item.o3,
                    }));
                    const dataNo2 = response.data.data.map((item) => ({
                        waktu: formatWaktu(item.created_at),
                        count: item.no2,
                    }));

                    dataPm10.reverse();
                    dataPm25.reverse();
                    dataSo2.reverse();
                    dataCo.reverse();
                    dataO3.reverse();
                    dataNo2.reverse();

                    // console.log("Pm10 Data:", dataPm10);
                    // console.log("Pm25 Data:", dataPm25);
                    // console.log("SO2 Data:", dataSo2);
                    // console.log("CO Data:", dataCo);
                    // console.log("O3 Data:", dataO3);
                    // console.log("NO2 Data:", dataNo2);

                    chartPm10?.destroy();
                    chartPm25?.destroy();
                    chartSo2?.destroy();
                    chartCo?.destroy();
                    chartO3?.destroy();
                    chartNo2?.destroy();

                    chartPm10 = createChart("pm10Chart", "PM10", dataPm10);
                    chartPm25 = createChart("pm25Chart", "PM25", dataPm25);
                    chartSo2 = createChart("so2Chart", "SO2", dataSo2);
                    chartCo = createChart("coChart", "CO", dataCo);
                    chartO3 = createChart("o3Chart", "O3", dataO3);
                    chartNo2 = createChart("no2Chart", "NO2", dataNo2);
                    // Extract data for each pollutant and dates
                }
            })
            .catch((error) => {
                console.log(error);
            });
    }

    hitBe();

    Echo.channel("data-channel").listen("NewDataInserted", (data) => {
        console.log(data.data);
        if (data.data.location == selectLocation.val()) {
            // pm10Elm.html(data.data.pm10);
            // pm2Elm.html(data.data.pm25);
            // so2Elm.html(data.data.so2);
            // coElm.html(data.data.co);
            // o3Elm.html(data.data.o3);
            // no2Elm.html(data.data.no2);
            // hasilPrediksiElm.html(categori[data.data.prediction_result]);
            // AkurasiElm.html(`${data.data.accuracy} %`);
            hitBe(selectLocation.val());
        }
    });

    selectLocation.select2();
    selectLocation.on("select2:select", function (e) {
        const locationId = $(this).val();
        hitBe(locationId);
    });
});
