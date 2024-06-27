img = getTitle();
dirPath = getMetadata("dirPath");

run("Split Channels");
selectWindow("C1-" + img);
close();
selectWindow("C2-" + img);

run("Select All");
setBackgroundColor(0, 0, 0);
 setForegroundColor(255, 255, 255);
run("Clear", "slice");

run("Convert to Mask");
run("Convert to Mask");
run("Grays");
saveAs("Tiff", replace(dirPath + img, "_CH[0-9].tif", "_mask.tif"));
close();
