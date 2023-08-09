import $ from "jquery";
window.$ = $;

$(function () {
    const pm10Elm = $("#pm10");
    const pm2Elm = $("#pm25");
    const so2Elm = $("#so2");
    const coElm = $("#co");
    const o3Elm = $("#o3");
    const no2Elm = $("#no2");
    const hasilPrediksiElm = $("#prediction");
    const AkurasiElm = $("#akurasi");

    const categori = ["BAIK", "SEDANG", "TIDAK SEHAT"];

    let config = {
        method: "get",
        maxBodyLength: Infinity,
        url: `${window.origin}/api/get-latest-prediction`,
        headers: {},
    };

    axios
        .request(config)
        .then((response) => {
            console.log(response.data);
            if (response.data.status) {
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
            }
        })
        .catch((error) => {
            console.log(error);
        });

    Echo.channel("data-channel").listen("NewDataInserted", (data) => {
        console.log(data.data);
        pm10Elm.html(data.data.pm10);
        pm2Elm.html(data.data.pm25);
        so2Elm.html(data.data.so2);
        coElm.html(data.data.co);
        o3Elm.html(data.data.o3);
        no2Elm.html(data.data.no2);
        hasilPrediksiElm.html(categori[data.data.prediction_result]);
        AkurasiElm.html(`${data.data.accuracy} %`);
        // You can make Axios requests here or do any other actions.
    });
});
