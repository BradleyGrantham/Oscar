function readURL(input) {
    var request;
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $("#result").text('');
            request = $.ajax({
                type: "POST",
                url: "http://54.216.29.159/oscar/",
                data: JSON.stringify({ b64string: e.target.result}),
                success: function(resultData){
                    $("#result").text(resultData[0].caption);
                },
                error: function(msg) {
                    console.log(msg);
                }
            });
            $('#myimg')
                .attr('src', e.target.result)
                .width(400);

        };

        reader.readAsDataURL(input.files[0]);
    }
}