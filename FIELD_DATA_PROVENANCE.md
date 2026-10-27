# Olympic Dam field data provenance

## Identified source chain

Zhang, Y. M., Xiong, S. Q., Wang, H., Wang, W. Y., and Yang, M. (2025). 3D physical property inversion for gravity data based on threshold pruning iterative algorithm. Chinese Journal of Geophysics (in Chinese), 68(9), 3616–3633. https://doi.org/10.6038/cjg2024S0470

Section4, printed page3629 (PDF page14), identifies SARIG (South Australian Resources Information Gateway) as the source of the Olympic Dam Bouguer gravity data. Figures18–21 on printed page3630 show the gravity, terrain, regional and residual maps of the same study area.

The four archived files od_grv.grd, od_dem.grd, grv_regC06S02.grd and grv_resC06S02.grd in the current project are byte-identical, by SHA-256, to corresponding files retained in the earlier study workspace. See provenance/field_source_trace.json. This establishes continuity with the published study, not a new download from the portal. The present author received these materials through colleagues; the original download transaction/date, product version and datum/EPSG record were not retained.

The cited article explicitly identifies SARIG for Bouguer gravity. The archived DEM is traceable to the same earlier project, but this alone does not identify its original acquisition dataset. No independent DEM product name is assigned.

## Official provider and access

Department for Energy and Mining, Government of South Australia; Geological Survey of South Australia. Geophysical data. https://energymining.sa.gov.au/industry/geological-survey/products-and-services/geophysical-data

SARIG map: https://map.sarig.sa.gov.au/

Official data services: https://energymining.sa.gov.au/industry/geological-survey/products-and-services/map-viewers-databases-and-services/data-services

These official pages confirm that South Australian gravity observations and statewide grids are made accessible through SARIG, with gravity stations and regional geophysical images as the relevant layers. They establish a current provider/access route. They do not establish which current statewide product is identical to the historical clipped grids.

Official pages and service capabilities were checked on27October2026. That date is not represented as the original data-download date.

## Exact numerical materials supplied

| File | Role |
|---|---|
| data/field_local/od_residual.grd | Actual residual gravity input to the continuation example |
| data/field_local/od_terrain.grd | Actual positive-up elevation grid on the gravity nodes |
| data/field_local/od_bouguer.grd | Archived original Bouguer grid on the continuation nodes |
| data/field_local/od_regional.grd | Archived regional field on the continuation nodes |
| data/field_local/source_grids/od_grv.grd | Earlier study Bouguer grid |
| data/field_local/source_grids/od_dem.grd | Earlier study terrain grid before interpolation to gravity nodes |
| data/field_local/source_grids/grv_regC06S02.grd | Earlier study regional product |
| data/field_local/source_grids/grv_resC06S02.grd | Earlier study residual product |
| reference_results/manuscript/real_*.grd | Historical fits, residual and four continued fields; not newly reproduced results |

The original Bouguer grid has100x94 nodes; easting671925–690744m, northing6621568–6638818m; gravity range-22.586305809715 to0.72266331190844mGal. The original terrain grid has76x100 nodes over the same extent and elevations84.541096023639–113.70977015138m. These values come from the archived DSAA headers. The example uses the supplied numerical coordinates; no EPSG code or geodetic transformation is inferred. A new spatial datum would require independent evidence.

Zhang et al.(2025) reports minimum-curvature regional–residual separation. The packaged continuation example starts from the archived processed regional/residual grids. It does not claim to regenerate that historical separation from a new portal download. The source scripts used in this continuation study resample the terrain onto gravity nodes; existing archived inputs are supplied unchanged.

## Attribution and rights

Original Bouguer data source attribution: Department for Energy and Mining, Government of South Australia, Geological Survey of South Australia, via SARIG, as documented by Zhang et al.(2025). Numerical clipping, prior processing and archived-file history are distinguished from the original data service.

The MIT License covers software only. Original third-party data rights remain with their respective holders. Provider copyright and reuse guidance: https://www.energymining.sa.gov.au/footer/copyright

The official copyright page permits reproduction subject to attribution, specific product conditions and third-party rights, and gives precedence to specific licenses. This is not evidence that these particular historical grids carry any specified Creative Commons version. No blanket MIT or CC BY license is assigned to the field materials, and no original provider rights are waived. The author has authorized sharing the retained research materials. This record documents attribution and known limitations; it is not an independent legal clearance of an unidentified product license.
