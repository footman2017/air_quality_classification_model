import $ from "jquery";
window.$ = $;

$(function () {
    // console.log();
    // setTimeout(() => {
    //     $("#mantap").html("hehehe");
    // }, 500);
    let config = {
        method: "get",
        maxBodyLength: Infinity,
        url: "http://127.0.0.1:8000/api/get-latest-prediction",
        headers: {},
    };

    axios
        .request(config)
        .then((response) => {
            console.log(JSON.stringify(response.data));
        })
        .catch((error) => {
            console.log(error);
        });

    Echo.channel("data-channel").listen("NewDataInserted", (data) => {
        console.log(data);
        // You can make Axios requests here or do any other actions.
    });
});
