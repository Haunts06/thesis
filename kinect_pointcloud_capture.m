%% kinect_pointcloud_capture.m
% Captures a single colorized point cloud from Kinect (v1 or v2),
% displays it, and saves to pointcloud_0001.ply, pointcloud_0002.ply, ...

function kinect_pointcloud_capture()
    clc; close all;

    %-- 0) Sanity checks
    assert(license('test','image_acquisition_toolbox')==1, ...
        'Image Acquisition Toolbox is required.');

    adaptors = imaqhwinfo;
    hasKinect = any(strcmpi({adaptors.InstalledAdaptors}, 'kinect'));
    assert(hasKinect, ['Kinect adaptor not found. ', ...
        'Install "Image Acquisition Toolbox Support Package for Kinect".']);

    kinfo = imaqhwinfo('kinect');
    if isempty(kinfo.DeviceInfo)
        error('No Kinect devices detected. Check USB/power and drivers.');
    end

    %-- 1) Open devices (DeviceIDs: 1=color, 2=depth for Kinect adaptor)
    % Choose available formats automatically.
    colorFmt = pickFormat(kinfo, 1, ["RGB","Color"]);
    depthFmt = pickFormat(kinfo, 2, ["Depth"]);

    % Use the System object interface (works well with pcfromkinect)
    colorDev = imaq.VideoDevice('kinect', 1, colorFmt);
    depthDev = imaq.VideoDevice('kinect', 2, depthFmt);

    cleanupObj = onCleanup(@() safeRelease(colorDev, depthDev));

    % Optional: auto white-balance/exposure can take a moment to settle
    warmupFrames = 20;
    for i=1:warmupFrames
        ~ = step(colorDev);
        ~ = step(depthDev);
    end

    %-- 2) Acquire synchronized frames
    colorImg = step(colorDev);   % uint8 RGB
    depthImg = step(depthDev);   % uint16 depth in mm

    %-- 3) Build colorized point cloud (uses Kinect intrinsics from depthDev)
    % pcfromkinect returns a pointCloud object with XYZ in meters
    ptCloud = pcfromkinect(depthDev, depthImg, colorImg);

    %-- 4) Visualize
    figure('Color','w','Name','Kinect Point Cloud');
    pcshow(ptCloud, 'MarkerSize', 45);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('Colorized Point Cloud from Kinect');

    %-- 5) Save with iterative filename
    outName = nextName('pointcloud_', '.ply');
    pcwrite(ptCloud, outName, 'Encoding','ascii'); % ascii = easy to inspect
    fprintf('✅ Saved point cloud to %s\n', outName);

    %-- 6) (Optional) Keep grabbing on keypress
    choice = questdlg('Capture another point cloud?', 'Repeat', 'Yes','No','No');
    while strcmp(choice,'Yes')
        % Re-acquire to get a fresh frame
        colorImg = step(colorDev);
        depthImg = step(depthDev);
        ptCloud  = pcfromkinect(depthDev, depthImg, colorImg);

        clf; pcshow(ptCloud, 'MarkerSize', 45);
        xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
        title('Colorized Point Cloud from Kinect');

        outName = nextName('pointcloud_', '.ply');
        pcwrite(ptCloud, outName, 'Encoding','ascii');
        fprintf('✅ Saved point cloud to %s\n', outName);

        choice = questdlg('Capture another point cloud?', 'Repeat', 'Yes','No','No');
    end
end

% -------- Helpers --------
function fmt = pickFormat(kinfo, deviceID, preferredKeywords)
    dev = kinfo.DeviceInfo([kinfo.DeviceInfo.DeviceID] == deviceID);
    fmts = string(dev.SupportedFormats);
    if isempty(fmts)
        error('No supported formats found for DeviceID %d.', deviceID);
    end
    % Try to find a format that contains one of the preferred keywords
    hit = [];
    for kw = preferredKeywords
        hit = [hit; find(contains(fmts, kw, 'IgnoreCase',true))]; %#ok<AGROW>
    end
    hit = unique(hit,'stable');
    if ~isempty(hit)
        fmt = char(fmts(hit(1)));
    else
        % Fall back to the first available format
        fmt = char(fmts(1));
    end
end

function safeRelease(colorDev, depthDev)
    try; release(colorDev); end %#ok<TRYNC>
    try; release(depthDev); end %#ok<TRYNC>
end

function name = nextName(prefix, ext)
    % Find next available prefix_XXXX.ext
    k = 1;
    while true
        name = sprintf('%s%04d%s', prefix, k, ext);
        if ~isfile(name), break; end
        k = k + 1;
    end
end
