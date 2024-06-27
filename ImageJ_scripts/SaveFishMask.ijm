img = getTitle();
dirPath = getMetadata("dirPath");
fileName = getMetadata("fileName");

for (slice = 1; slice <= 2; slice ++) {
    setSlice(slice);
    setBackgroundColor(0, 0, 0);
    run("Clear Outside");
    setForegroundColor(255, 255, 255);
    run("Fill", "slice");
}

run("Split Channels");
selectWindow("C1-" + img);
close();
selectWindow("C2-" + img);

setOption("BlackBackground", true);
run("Convert to Mask");
run("Grays");
saveAs("Tiff", replace(dirPath + img, "_CH[0-9].tif", "_mask.tif"));
