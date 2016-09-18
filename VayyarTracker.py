from __future__ import division
from collections import namedtuple
from math import pi
import ctypes as ct
import numpy as np
try:
    range = xrange
except NameError:
    pass


def initTracker(wlbt):

    def _initCfg():
        imgProcessing = {
            "SmoothImageWindow": np.array([15/180*pi, 30/180*pi, 0.15])
        }
        Tracker = {
            "numTrackedTargets": 4,
            "NormalizeNoiseLevel": True,
            "DetectionThresholds": np.array([2, 2, 1, 0, -4, 4, -3]),
            "RmsAccelarationPerSec": 10,
            "MaxVelocity": 20 * 3.6,
            "IntensityAvgWinDuration": 3,
            "AssumedPeakResolution": np.array([15/180*pi, 6/180*pi, 0.1]) + imgProcessing["SmoothImageWindow"],
            "AssumedStainResolution": np.array([30/180*pi, 20/180*pi, 0.2]) / 2,
            "AssumedStainFloordB": np.array([-10, -10, -25, -25, -25, -15]),
            "TreatImageAsLLR": False,
            "LocationDisplaySmoothingTime": 0,
            "NoiseLevelIntegrationTime": 10,
            "BabiesCompetitionFactor": np.array([1.5, 6]),
            "DisregardImageThreshold": 1,
            "UseParabolicApproximation": True,
            "ArenaEntryExitEdges": np.array([0, 0, 1, 1, 0, 1]),
            "SpatialNoiseLevel": False
        }
        Common = {
            "AssumedFrameRate": 6,
        }
        Cfg = {
            "imgProcessing": imgProcessing,
            "Common": Common,
            "Tracker": Tracker
        }
        return Cfg

    def _initState():
        Nt = Cfg["Tracker"]["numTrackedTargets"]
        Targets = {
            "isValid": np.zeros(Nt, dtype=np.bool),
            "isLocked": np.zeros(Nt, dtype=np.bool),
            "Location": np.empty((Nt, 3)) * np.nan,
            "KState": np.empty((2, 3, Nt)) * np.nan,
            "KCov": np.empty((2, 2, 3, Nt)) * np.nan,
            "DbgPrediction": np.empty((Nt, 6)) * np.nan,
            "CurrIntensity": np.empty(Nt) * np.nan,
            "AvgIntensity": np.empty(Nt) * np.nan,
            "LifeAccumulator": np.empty(Nt) * np.nan,
            "ID": np.arange(1, Nt+1, dtype=np.uint8)
        }
        tState = {
            "NumTargets": Nt,
            "Targets": Targets,
            "PreviousTime": np.nan,
            "EventCounts": np.zeros(9, dtype=np.uint8),
            "ImgNoiseLevelEst": 0,
            "ImgNoiseLevelAccumTime": 0,
            "externNoiseLevelEstRms": np.array([])
        }
        return tState

    def _initImgData(theta, phi, r):
        img_data = {
            "xi_grid": np.r_[theta[0] : theta[1]+1 : theta[2]],
            "yi_grid": np.r_[phi[0]   : phi[1]+1   : phi[2]],
            "zi_grid": np.r_[r[0]     : r[1]+1     : r[2]],
            "axis_type": 2
        }
        return img_data

    global Cfg, tState, img_data
    Cfg = _initCfg()
    tState = _initState()
    img_data = _initImgData(
        wlbt.GetArenaTheta(),
        wlbt.GetArenaPhi(),
        wlbt.GetArenaR()
    )


Cfg = None
tState = None
img_data = None

TrackerTarget = namedtuple('SensorTarget', 'xPosCm yPosCm zPosCm ID')

def getTrackerState(wlbt, numpy=True):
    global tState, img_data
    if not img_data:
        raise Exception("Please 'init()' before using the tracker.")
    img_data["images"] = _getUnfilteredImage(wlbt)
    if numpy:
        kTargets =  _kalmanFilter(img_data, tState, Cfg)["Targets"]
        logic = kTargets["isValid"] & kTargets["isLocked"]
        idMat = kTargets["ID"][logic]
        locMat = kTargets["Location"][logic]
        return [TrackerTarget(
            locMat[i][0],
            locMat[i][1],
            locMat[i][2],
            idMat[i]
        ) for i in range(idMat.size)]
    else:
        return _kalmanFilter(img_data, tState, Cfg)

def _getUnfilteredImage(wlbt):

    def _getRawImageNumpy(wlbt):
        rasterImage = ct.POINTER(ct.c_int)()
        sizeX, sizeY, sizeZ = ct.c_int(), ct.c_int(), ct.c_int()
        power = ct.c_double()
        res = wlbt._wlbt.Walabot_GetRawImage(ct.byref(rasterImage),
            ct.byref(sizeX), ct.byref(sizeY), ct.byref(sizeZ), ct.byref(power))
        array = ct.c_int * sizeX.value * sizeY.value * sizeZ.value
        addr = ct.addressof(rasterImage.contents)
        rasterImage = np.frombuffer(array.from_address(addr), dtype=np.intc)
        rasterImage = rasterImage.reshape(sizeZ.value, sizeY.value, sizeX.value)
        rasterImage = rasterImage.transpose()
        return rasterImage, power.value

    def _unfilterImage(rawImage, power):
        return rawImage * power / 100 / 255

    return _unfilterImage(*_getRawImageNumpy(wlbt))


def _kalmanFilter(img_data, tState, Cfg):

    targets = tState["Targets"] # fetch

    flag_isDataPolar = (img_data["axis_type"] == 2)
    tState["axis_type"] = img_data["axis_type"]

    # Give names to detection thresholds (as they are so many..)
    T_search = Cfg["Tracker"]["DetectionThresholds"][0]
    T_ss = Cfg["Tracker"]["DetectionThresholds"][1]
    T_lock = Cfg["Tracker"]["DetectionThresholds"][2]
    T_unlock = Cfg["Tracker"]["DetectionThresholds"][3]
    T_L = Cfg["Tracker"]["DetectionThresholds"][4]
    T_H = Cfg["Tracker"]["DetectionThresholds"][5]
    T_edge = Cfg["Tracker"]["DetectionThresholds"][6]

    validTargets = np.reshape(np.argwhere(targets["isValid"]), -1, order="F")

    dT = 1 / Cfg["Common"]["AssumedFrameRate"]
    gridStepSize = np.array([
        _getGridStepSize(img_data["xi_grid"]),
        _getGridStepSize(img_data["yi_grid"]),
        _getGridStepSize(img_data["zi_grid"])
    ])
    gridOffset = np.array([
        img_data["xi_grid"][0],
        img_data["yi_grid"][0],
        img_data["zi_grid"][0]
    ])
    gridSize = np.array([
        img_data["xi_grid"].size,
        img_data["yi_grid"].size,
        img_data["zi_grid"].size
    ])
    img = np.absolute(img_data["images"])
    if np.all(np.isnan(img.flatten(order="F"))):
        return

    # noise estimation and normalization
    if tState["ImgNoiseLevelAccumTime"] == 0: # init
        tState["ImgNoiseLevelEst"] = np.mean(
            np.absolute(img.flatten(order="F"))**2
        )
        if Cfg["Tracker"]["SpatialNoiseLevel"]:
            tState["ImgNoiseLevelEst"] = np.dot(
                tState["ImgNoiseLevelEst"],
                np.ones(img.size)
            )
    img_target_induced_noise = np.zeros((
        img.shape[0],
        img.shape[1],
        img.shape[2],
        max(targets["isValid"].shape)))

    if tState["externNoiseLevelEstRms"].size: # not empty
        extern_noise_level = tState["externNoiseLevelEstRms"]["images"]
    else:
        extern_noise_level = np.zeros(img.shape)
    ImgNoiseLevel = tState["ImgNoiseLevelEst"] ** 0.5

    for ti in validTargets:

        VelocityVarInc = np.array([1, 1, 1]) * Cfg["Tracker"]["RmsAccelarationPerSec"]**2 * dT**2
        if flag_isDataPolar:
            Location = targets["KState"][0, :, ti]
            VelocityVarInc[:2] = VelocityVarInc[:2] / Location[2]**2

        StateUpdateMatrix = np.array([[1, dT], [0, 1]])
        StateNoiseCov = np.array([[dT**2/4, dT/2], [dT/2, 1]])

        for dim in range(3):
            targets["KState"][:, dim, ti] = np.dot(
                StateUpdateMatrix, targets["KState"][:, dim, ti]
            )
            targets["KCov"][:, :, dim, ti] = np.dot(
                np.dot(
                    StateUpdateMatrix,
                    targets["KCov"][:, :, dim, ti]
                ), StateUpdateMatrix.conj().T
            ) + np.dot(StateNoiseCov, VelocityVarInc[dim])
            targets["KCov"][0, 0, dim, ti] += gridStepSize[dim]**2

        Loc = targets["KState"][0, :, ti]
        isOutOfArenaEdge = np.array([
            Loc[0] < img_data["xi_grid"][0],
            Loc[0] > img_data["xi_grid"][-1],
            Loc[1] < img_data["yi_grid"][0],
            Loc[1] > img_data["yi_grid"][-1],
            Loc[2] < img_data["zi_grid"][0],
            Loc[2] > img_data["zi_grid"][-1]
        ])

        if np.any(isOutOfArenaEdge & Cfg["Tracker"]["ArenaEntryExitEdges"]):
            targets["isLocked"][ti] = False
            targets["isValid"][ti] = False
            tState["EventCounts"][7] = tState["EventCounts"][7] + 1
            continue # BUG: what is it for?

        if np.any(np.isnan(targets["KState"][:, :, ti])):
            raise Exception('kalmanFilter - internal')

        targets["DbgPrediction"][ti, :] = np.concatenate([
            targets["KState"][0, :, ti],
            np.reshape(targets["KCov"][0, 0, :, ti]**0.5, -1, order="F")
        ])

        img_target_induced_noise[:, :, :, ti] = targets["AvgIntensity"][ti] * _trackerGetStainMask(
                img_data, targets["KState"][0, :, ti],
                Cfg["Tracker"]["AssumedStainResolution"],
                Cfg["Tracker"]["AssumedStainFloordB"]
            )

    validTargets = np.reshape(np.argwhere(targets["isValid"]), -1, order="F")

    for ti in validTargets.T:

        rng = targets["isValid"] & targets["isLocked"] & np.not_equal(np.arange(targets["isValid"].size), ti)

        img_target_mask_level = extern_noise_level

        if np.any(rng):
            img_target_mask_level = np.maximum(
                img_target_mask_level,
                img_target_induced_noise[:, :, :, rng].max(3)
            )
        img_substracted = np.maximum(img - img_target_mask_level, 0)
        if Cfg["Tracker"]["NormalizeNoiseLevel"]:
            img_normalized = img_substracted / np.maximum(img_target_mask_level, ImgNoiseLevel)
        else:
            img_normalized = img_substracted

        # produce a masked image acccoring to the prior on the target
        img_mask_x = 0.5 * (targets["KState"][0, 0, ti] - img_data["xi_grid"]) **2 / targets["KCov"][0, 0, 0, ti]
        img_mask_y = 0.5 * (targets["KState"][0, 1, ti] - img_data["yi_grid"]) **2 / targets["KCov"][0, 0, 1, ti]
        img_mask_z = 0.5 * (targets["KState"][0, 2, ti] - img_data["zi_grid"]) **2 / targets["KCov"][0, 0, 2, ti]

        # Combine mask with image
        if Cfg["Tracker"]["TreatImageAsLLR"]:
            img_LLR = img_normalized ** 2
            img_masked = img_LLR - img_mask_x - img_mask_y - img_mask_z
        else:
            normdist = img_mask_x[:, None, None] + img_mask_y[None, :, None] + img_mask_z[None, None, :]
            mask = 1 / (1 + normdist + 0.5 * normdist**2)
            img_masked = img_normalized * mask

        # find peak of masked image
        ix = np.argmax(img_masked.flatten(order="F"))
        img_peak = img_masked.flatten(order="F")[ix]
        xi, yi, zi = np.unravel_index(ix, img.shape, order="F")
        x, y, z = img_data["xi_grid"][xi], img_data["yi_grid"][yi], img_data["zi_grid"][zi]
        Location = np.array([x, y, z])
        img_peak_unmasked = img.flatten(order="F")[ix]
        img_at_peak_normalized = img_normalized.flatten(order="F")[ix]

        if Cfg["Tracker"]["UseParabolicApproximation"]: # correct the location relative to the peak.
            Location = Location + _matrixPeakParabolicFix(
                img_masked,
                np.array([xi, yi, zi])
            ) * gridStepSize

        # kalman state update
        if Cfg["Tracker"]["TreatImageAsLLR"] or img_at_peak_normalized > Cfg["Tracker"]["DisregardImageThreshold"]:
            AssumedPeakResolutionSqr = Cfg["Tracker"]["AssumedPeakResolution"] ** 2
            if Cfg["Tracker"]["TreatImageAsLLR"]:
                AssumedPeakResolutionSqr /= img_at_peak_normalized + 0.01
            for dim in range(3):
                Kn_gain = targets["KCov"][:, 0, dim, ti] / (targets["KCov"][0, 0, dim, ti] + AssumedPeakResolutionSqr[dim])
                targets["KState"][:, dim, ti] += Kn_gain / Kn_gain[0] * (Location[dim] - targets["KState"][0, dim, ti])
                targets["KCov"][:, :, dim, ti] -= Kn_gain[:, None] * targets["KCov"][0, :, dim, ti]

        # update CurrIntensity, and AvgIntensity using simple IIR
        targets["CurrIntensity"][ti] = img_peak_unmasked
        targets["AvgIntensity"][ti] += dT/Cfg["Tracker"]["IntensityAvgWinDuration"] * (img_peak_unmasked - targets["AvgIntensity"][ti])

        targets["Location"][ti, :] += np.dot(
                np.minimum(dT / (Cfg["Tracker"]["LocationDisplaySmoothingTime"] + np.spacing(1)), 1),
                targets["KState"][0, :, ti] - targets["Location"][ti, :]
            )

        if np.any(np.any(np.isnan(targets["KState"][:, :, ti]))):
            raise('internal')

        targets["LifeAccumulator"][ti] = np.minimum(
            targets["LifeAccumulator"][ti] + np.dot(dT, (img_at_peak_normalized - T_ss)),
            T_H
        )

        if targets["LifeAccumulator"][ti] >= T_lock and (not targets["isLocked"][ti]):
            targets["isLocked"][ti] = True
            tState["EventCounts"][0] = tState["EventCounts"][0] + 1
        if targets["LifeAccumulator"][ti] < T_unlock:
            targets["isLocked"][ti] = False
        if targets["LifeAccumulator"][ti] <= T_L:
            targets["isLocked"][ti] = False
            targets["isValid"][ti] = False
            tState["EventCounts"][1] = tState["EventCounts"][1] + 1

        if targets["isValid"][ti]:
            img_target_induced_noise[:, :, :, ti] = np.absolute(img_peak_unmasked) * 1.2 * _trackerGetStainMask(
                img_data, targets["KState"][0, :, ti],
                Cfg["Tracker"]["AssumedStainResolution"],
                Cfg["Tracker"]["AssumedStainFloordB"]
            )

    # update noise level
    points_not_hidden_by_targets = ~np.isnan(img) & (np.absolute(img) > 3 * np.maximum(img_target_induced_noise.max(3), extern_noise_level))
    if np.any(points_not_hidden_by_targets.flatten(order="F")):
        tState["ImgNoiseLevelAccumTime"] = np.minimum(
            tState["ImgNoiseLevelAccumTime"] + dT,
            Cfg["Tracker"]["NoiseLevelIntegrationTime"]
        )
        if Cfg["Tracker"]["SpatialNoiseLevel"]:
            tState["ImgNoiseLevelEst"] += np.dot(
                np.linalg.solve(dT, tState["ImgNoiseLevelAccumTime"]),
                points_not_hidden_by_targets
            ) * (np.absolute(img)**2 - tState["ImgNoiseLevelEst"])
        else:
            updateFactor = np.sum(points_not_hidden_by_targets.flatten(order="F")) / points_not_hidden_by_targets.size
            imageSqr = np.absolute(img[points_not_hidden_by_targets])**2
            imagePwr = np.mean(imageSqr)
            NoiseThisImage = np.mean(imageSqr[imageSqr < 4 * imagePwr])
            tState["ImgNoiseLevelEst"] += updateFactor * dT / tState["ImgNoiseLevelAccumTime"] * (NoiseThisImage - tState["ImgNoiseLevelEst"])

    img_target_mask_level = extern_noise_level
    if np.any(targets["isValid"]):
        img_target_mask_level += np.amax(img_target_induced_noise[:, :, :, targets["isValid"]], axis=3)

    img_normalized = np.maximum(img - img_target_mask_level, 0)
    ImgNoiseLevel_norm = ImgNoiseLevel
    if Cfg["Tracker"]["NormalizeNoiseLevel"]:
        img_normalized /= np.maximum(img_target_mask_level, ImgNoiseLevel)
        ImgNoiseLevel_norm = 1

    ix = np.argmax(img_normalized.flatten(order="F"))
    img_peak = img_normalized.flatten(order="F")[ix]
    img_peak_scaled = img_peak

    if Cfg["Tracker"]["TreatImageAsLLR"]:
        img_peak_scaled = img_peak_scaled ** 2
    img_peak_unscaled = img.flatten(order="F")[ix]

    if img_peak_scaled >= T_search:
        tState["EventCounts"][2] = tState["EventCounts"][2] + 1
        xi, yi, zi = np.unravel_index(ix, img.shape, order='F')
        x = img_data["xi_grid"][xi]
        y = img_data["yi_grid"][yi]
        z = img_data["zi_grid"][zi]
        img_thresh = (2 * targets["isValid"]).astype(np.float64)
        unlocked_rng = targets["isValid"] & -targets["isLocked"]
        locked_rng = targets["isValid"] & targets["isLocked"]
        invalid_with_history_rng = -targets["isValid"] & -np.isnan(targets["Location"][:, 0])
        img_thresh[unlocked_rng] = targets["CurrIntensity"][unlocked_rng] * Cfg["Tracker"]["BabiesCompetitionFactor"][0]
        img_thresh[locked_rng] = np.maximum(
            targets["CurrIntensity"][locked_rng],
            targets["AvgIntensity"][locked_rng]
        ) * Cfg["Tracker"]["BabiesCompetitionFactor"][1]
        img_thresh[invalid_with_history_rng] = -1 / (np.sum((targets["KState"][0, :, invalid_with_history_rng] - np.array([x, y, z]))**2, 1) + 1)

        img_thresh_min, ti = np.amin(img_thresh), np.argmin(img_thresh)

        if img_peak > img_thresh_min:
            event_index = 3 + 2*targets["isValid"][ti] + targets["isLocked"][ti]
            tState["EventCounts"][event_index] = tState["EventCounts"][event_index] + 1

            # replace the target
            targets["isValid"][ti] = True
            targets["isLocked"][ti] = False
            targets["Location"][ti, :] = np.array([x, y, z])
            targets["KState"][:, :, ti] = np.array([[x, y, z], [0, 0, 0]])

            MaxVelocityPerDim = np.array([1, 1, 1], dtype=np.float64) * Cfg["Tracker"]["MaxVelocity"]
            if flag_isDataPolar:
                MaxVelocityPerDim[0:2] = MaxVelocityPerDim[0:2] / z

            for dim in range(3):
                targets["KCov"][:, :, dim, ti] = np.diag([
                    Cfg["Tracker"]["AssumedPeakResolution"][dim]**2,
                    MaxVelocityPerDim[dim]**2
                ])

            targets["CurrIntensity"][ti] = img_peak_unscaled
            targets["AvgIntensity"][ti] = img_peak_unscaled

            arenaEdges = np.array([
                [img_data["xi_grid"][0], img_data["yi_grid"][0], img_data["zi_grid"][0]],
                [img_data["xi_grid"][-1], img_data["yi_grid"][-1], img_data["zi_grid"][-1]]
            ])
            minDistanceFromArenaEdge = np.absolute(targets["Location"][ti, :] - arenaEdges)
            minDistanceFromArenaEdge[:, np.array([
                img_data["xi_grid"].size,
                img_data["yi_grid"].size,
                img_data["zi_grid"].size]) <= 1] = np.inf # BUG: VisibleDeprecationWarning
            isArenaEdge = minDistanceFromArenaEdge < MaxVelocityPerDim * dT + Cfg["Tracker"]["AssumedPeakResolution"]
            isArenaEdge = isArenaEdge * np.reshape(Cfg["Tracker"]["ArenaEntryExitEdges"], (2, 3), order="F")
            if np.any(isArenaEdge.flatten(order="F")):
                LifeAccum0 = T_edge
                tState["EventCounts"][8] = tState["EventCounts"][8] + 1
            else:
                LifeAccum0 = T_L

            targets["LifeAccumulator"][ti] = np.minimum(
                LifeAccum0 + dT * (img_peak_scaled - T_ss),
                T_H
            )
            targets["DbgPrediction"][ti, :] = np.concatenate([
                targets["KState"][0, :, ti],
                np.reshape(np.sqrt(targets["KCov"][0, 0, :, ti]), -1, order="F")
            ])

        else:
            tState.EventCounts[4] = tState.EventCounts[4] + 1

    # sort targets by intensity (important for order of peeling, next time)
    sort_key = np.zeros((targets["isValid"].shape))
    sort_key[targets["isValid"]] = targets["AvgIntensity"][targets["isValid"]]
    sort_key[targets["isLocked"]] += np.max(targets["AvgIntensity"])
    sort_ix = np.argsort(-sort_key)
    targets["isValid"] = targets["isValid"][sort_ix]
    targets["isLocked"] = targets["isLocked"][sort_ix]
    targets["Location"] = targets["Location"][sort_ix, :]
    targets["KState"] = targets["KState"][:, :, sort_ix]
    targets["KCov"] = targets["KCov"][:, :, :, sort_ix]
    targets["AvgIntensity"] = targets["AvgIntensity"][sort_ix]
    targets["CurrIntensity"] = targets["CurrIntensity"][sort_ix]
    targets["LifeAccumulator"] = targets["LifeAccumulator"][sort_ix]
    targets["ID"] = targets["ID"][sort_ix]
    targets["DbgPrediction"] = targets["DbgPrediction"][sort_ix, :]

    targets["LifeAccumulator"][~targets["isValid"]] = np.nan

    tState["Targets"] = targets

    return tState


def _getGridStepSize(lin_grid):
    if lin_grid.size < 2:
        return 0
    return (lin_grid[-1] - lin_grid[0]) / (lin_grid.size - 1)

def _trackerGetStainMask(img_data, Location, AssumedStainResolution, AssumedStainFloordB):
    floorLin = 10 ** (AssumedStainFloordB / 20)
    dX = (img_data["xi_grid"][:] - Location[0]) / AssumedStainResolution[0]
    img_mask_x = np.transpose(
        np.maximum(
            1 / (1 + dX**2)**0.5,
            np.reshape(floorLin[0 + (dX > 0)], -1, order="F")
        )[:, None, None],
        (0, 1, 2)
    )
    dY = (img_data["yi_grid"][:] - Location[1]) / AssumedStainResolution[1]
    img_mask_y = np.transpose(
        np.maximum(
            1 / (1 + dY**2)**0.5,
            np.reshape(floorLin[2 + (dY > 0)], -1, order="F")
        )[:, None, None],
        (1, 0 ,2)
    )
    dZ = (img_data["zi_grid"][:] - Location[2]) / AssumedStainResolution[2]
    img_mask_z = np.transpose(
        np.maximum(
            1 / (1 + dZ**2)**0.5,
            np.reshape(floorLin[4 + (dZ > 0)], -1, order="F")
        )[:, None, None],
        (1, 2, 0)
    )
    return img_mask_x * img_mask_y * img_mask_z

def _matrixPeakParabolicFix(img_masked, xyzi):
    Habc = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])
    Labc = np.array([[0.5, -1, 0.5], [-0.5, 0, 0.5], [0, 1, 0]])
    location_offset = np.zeros(3)
    for d in range(3):
        if xyzi[d] > 0 and xyzi[d] < img_masked.shape[d] - 1:
            img_samples = np.zeros(3)
            img_samples[0] = img_masked[xyzi[0] - (d==0), xyzi[1] - (d==1), xyzi[2] - (d==2)]
            img_samples[1] = img_masked[xyzi[0]         , xyzi[1]         , xyzi[2]]
            img_samples[2] = img_masked[xyzi[0] + (d==0), xyzi[1] + (d==1), xyzi[2] + (d==2)]
            parabole_abc = np.dot(Labc, img_samples)
            location_offset[d] = -parabole_abc[1] / (2*parabole_abc[0]+np.spacing(1))
    return location_offset
