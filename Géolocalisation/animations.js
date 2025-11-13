var diff
var time = $("#timeline_time")
var timetracker = $("#timelabel")
var trackingjson
var mapcell
var dicocircle = []
var fromage, polycells
var playing = false
var listeantennes = []
$('.date').datepicker({
    format: 'yyyy-mm-dd'
});

$("a.usecase").on("click", function () {
    genMenu($(this).attr("id"))
});

$("#back_results").on("click", function(){
    layerGroup.clearLayers()
    $("#menu").slideToggle("slow")
    $("#results_realtime").slideToggle("slow")
})

function genMenu(id) {
    var shapebtn, menutemplate
    $("#menu").empty()
    if (id === "Visualisation") {
        menutemplate = $("#realtime_tp").html()
        $("#menu").append(menutemplate)
    } else {
        menutemplate = $("#useCase_tp").html()
        $("#menu").append(menutemplate)

        $("#titre").text(id)

        switch (id) {
            case "Trajet":
                shapebtn = $("#ligne_tp").html()
                break;

            case "Visite":
                shapebtn = $("#polybtn_tp").html()
                break;
            case "Habitude":
                shapebtn = $("#polybtn_tp").html()
                break;
        }
        $("#shapeBtnSlot").append(shapebtn)
    }

    $("#menu_usecase").slideUp("slow")
    $("#menu").slideToggle("slow");
    refreshEvents()
}






function parseTime(min) {
    var h = Math.floor(min / 3600)
    var m = Math.floor(min / 60) % 60
    var s = min - h * 3600 - m * 60
    var heure = timeToString([h, m, s])
    timetracker.text(heure)
}

function timeToString(x) {
    for (i in x) {
        if (x[i] < 10) {
            x[i] = "0" + x[i].toString()
        } else {
            x[i] = x[i].toString()
        }
    }
    return x.join(" : ")
}

function refreshEvents() {
    $(".back").on("click", goBack);

    $("#backRealtime").on("click", function () {
        clearInterval(fromage)
    });

    $("#play_button").on("click", togglePlay);

    $("#timeline_time").on("input", iterate);

    $("#submit_button").on("click", function () {
        var imei = $("#imei_realtime").val()
        var date = $("#date_realtime").val()
        GetTrackingData(imei, date)
    });

    $("#ligne").on("click", function () {
        pmDraw("Line")
    });

    $("#polygone").on("click", function () {
        pmDraw("Polygon")
    });

}


function togglePlay() {
    if (playing != true) {
        var current
        $("#play_button").text("Pause")
        fromage = setInterval(function () {
            current = parseInt($("#timeline_time").val())
            $("#timeline_time").val(current + 120)
            iterate()
        }, 100)
        playing = true
    } else {
        clearInterval(fromage)
        playing = false
        $("#play_button").text("Play")
    }
}


function iterate() {
    layerGroup.setStyle(realtimestyle)
    parseTime(time.val())
}

function goBack() {
    mymap.pm.disableDraw()
    layerGroup.clearLayers()
    $("#menu_usecase").slideDown("slow")
    $("#menu").slideUp("slow")
}

function pmDraw(shape) {
    layerGroup.clearLayers()
    if ($(this).hasClass("active")) {
        mymap.pm.disableDraw(shape)
    } else {
        mymap.pm.enableDraw(shape)
    }
}