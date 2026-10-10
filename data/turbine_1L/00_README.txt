This README file was generated on 2025-12-14 by Johannes Næsset Hillestad.
Last updated: 2026-01-11.


-------------------
GENERAL INFORMATION
-------------------
// Title of Dataset: Volumetric three-dimensional experimental measurement of a model wind turbine
// DOI: https://doi.org/10.18710/SX0ZQT
// Contact Information
     // Name: Worth, Nicholas A.
     // Institution: Norwegian University of Science and Technology (NTNU)
     // Email: nicholas.a.worth@ntnu.no
     // ORCID: https://orcid.org/0000-0002-7084-9304

// Contributors: See metadata field Contributor.
// Data Type: See metadata field Data Type.
// Date of Collection: See metadata field Date of Collection.
// Geographic location: See metadata section Geospatial Metadata.
// Funding sources: See metadata section Funding Information.

// Description of dataset: 
This dataset contains experimental volumetric velocimetry results obtained in the near wake of a model wind turbine (WT). The time-resolved velocity fields were obtained using the particle tracking velocimetry (PTV) technique Shake-the-Box (STB) and subsequent Lagrangian to Eulerian conversion using the Vortex-in-Cell# (VIC#) approach. Data for four cases are available. The four cases differ in freestream turbulence level and number of WTs present in the flow, see further details in ''METHODOLOGICAL INFORMATION'' below. The cases available in this dataset support the analysis in the Related Publication. Related Publication [1] analyzes case 1L (folder name 1WT_rand_100_2D.zip). 

[1] Hillestad, J. N., Yadala, S., Li, L., Hearst, R. J., Worth, N. A. (2026). Volumetric three-dimensional experimental measurement of vortex dynamics in a rotating wake. Physical Review Fluids, 11(1), 014701. https://doi.org/10.1103/9pwb-ky8n
 

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

// Description of sources and methods used for collection/generation of data: 
The data was collected in a large, optically accessible water channel at the Norwegian University of Science and Technology (NTNU). The channel test section measured 1.8 metres wide by 1 metre high by 11 metres long. For all four cases, the water level in the channel was 500 mm. The 45mm diametre, free-spinning WT whose near wake was being measured, was mounted inverted to a acrylic mounting plate lowered into the water. For two of the available cases, 1L and 1H, this was the only WT in the flow. For the other two cases, 2L and 2H, a second WT was present in the flow. The second WT was mounted 300 mm (~6.67D) upstream of the WT whose wake was being measured. In all four cases, the WTs were moved into the free stream by mounting them to extenders that resulted in a total hub height of 160 mm for both WTs. In all four cases, the WT hub centrelines were parallell with the mean flow direction and the channel walls. 

The freestream flow conditions were measured using laser Doppler velocimetry (LDV) at a distance of 5D upstream of the most upstream WT. The wake behind the WT itself was illuminated using eight GS Vitex MultiLED QT LED heads (8x150W) and imaged using four Photron FASTCAM Mini WX100 high-speed cameras. Seeding was obtained by using 40 micron polystyrene particles. Each of the cameras were equipped with Sigma 180mm F/3.5 EX APO Macro DG HSM lenses operated at f/#=16. Triggering and synchronisation was achieved using a LaVision PTU X controller. The cameras were operated at 500 Hz for a total of 5457 exposures, i.e. a sampling time of 10.9 seconds.


// Methods for processing the data: 
The raw images were processed in DaVis 10.1 to obtain the velocity fields. Following image preprocessing and subsequent volume self-calibration, the STB algorithm was used to obtain particle tracks. Finally, conversion from Lagrangian particle tracks to Eulerian fields was achieved using the vortex-in-cell algorithm, specifically VIC#.


// Experimental conditions:
Experiments were conducted at a freestream velocity of U=0.4 m/s. For cases 1L and 2L, the turbulence intensity of the incoming freestream was 5%. For cases 1H and 2H, the turbulence intensity of the incoming freestream was 10%. The turbulence intensity was measured by the LDV upstream of the WT and is defined as: sigma_u/U, where sigma_u is the standard deviation of the velocity perturbation of the streamwise velocity and U is the mean value of the streamwise velocity. The resulting freestream Reynolds' number, defined as Re=(UD)/nu, where U is the mean value of the freestream velocity, D is the rotor diameter and nu is the kinematic viscosity of water, was approximately 2e4. 


--------------------
DATA & FILE OVERVIEW
--------------------
// File List: 
	// 00_README.txt:
	// 1WT_rand_100_2D.zip: Contains the data for case 1L. See list of files in folder below. To unpack the file, about 225 GB of disk space is required.
		// instantaneous: Folder that contains the data for each time step (from time step 1 to time step 5457).
			// B*****.txt
		// rotor_angle.txt
		// scale_quantities.txt

	// 1WT_rand_005_2D.zip: Contains the data for case 1H. See list of files in folder below. To unpack the file, about 225 GB of disk space is required.
		// instantaneous: Folder that contains the data for each time step (from time step 1 to time step 5457).
			// B*****.txt
		// rotor_angle.txt
		// scale_quantities.txt

	// 2WT_rand_100_2D.zip: Contains the data for case 2L. See list of files in folder below. To unpack the file, about 225 GB of disk space is required.
		// instantaneous: Folder that contains the data for each time step (from time step 1 to time step 5457).
			// B*****.txt
		// rotor_angle.txt
		// scale_quantities.txt

	// 2WT_rand_005_2D.zip: Contains the data for case 2H. See list of files in folder below. To unpack the file, about 225 GB of disk space is required.
		// instantaneous: Folder that contains the data for each time step (from time step 1 to time step 5457).
			// B*****.txt
		// rotor_angle.txt
		// scale_quantities.txt


// Is this an updated version of a dataset published on DataverseNO? No.

-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: case_name.zip/instantaneous/B*****.txt, where case_name is any of the four folder case names, and ***** is any number between 00001 and 05457. 
-----------------------------------------
// Variable/Column List: 
	//      x: x-coordinate of the measurement. Unit: [mm].
	//      y: y-coordinate of the measurement. Unit: [mm].
	//	z: z-coordinate of the measurement. Unit: [mm].
	//      u: x-component of velocity at the position triplet in the same row. Unit: [m/s].
	// 	v: y-component of velocity at the position triplet in the same row. Unit: [m/s].
	//	w: z-component of velocity at the position triplet in the same row. Unit: [m/s].

-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: case_name.zip/rotor_angle.txt, where case_name is any of the four folder case names.
-----------------------------------------
// Variable/Column List: 
	//   time_step_vec: time step of measurement. Unit: [1].
	// rotor_angle_deg: angle of one specific rotor blade. Rotor blade aligned with positive z-axis is 0 degrees. Unit: [degree].

-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: case_name.zip/scale_quantities.txt, where case_name is any of the four folder case names.
-----------------------------------------
// Variable/Column List: 
	//     D: WT diametre. Unit: [mm].
	// U_hub: freestream velocity. Unit: [m/s].


--------------------------
SHARING/ACCESS INFORMATION
--------------------------
// Licenses/Restrictions: See Terms tab.
// Links to publications that cite or use the data: See metadata field Related Publication.
// Links/relationships to related data sets: See metadata field Related Dataset.
// Data sources: See metadata field Data Source.
// Recommended citation: See citation generated by repository.

