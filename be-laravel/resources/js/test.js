import Swal from "sweetalert2";

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
        // You can make Axios requests here or do any other actions.
    });

    selectLocation.select2();
    selectLocation.on("select2:select", function (e) {
        const locationId = $(this).val();
        hitBe(locationId);
    });
});
