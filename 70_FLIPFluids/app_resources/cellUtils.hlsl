#ifndef _FLIP_EXAMPLE_CELL_UTILS_HLSL
#define _FLIP_EXAMPLE_CELL_UTILS_HLSL

// Use when the "enum gets accidentally bitcasted to float" DXC bug is gone
#if 0
enum/* class*/ CellMaterial : uint16_t
{
    CM_SOLID = 0,
    CM_FLUID = 1,
    CM_AIR = 2
};
#endif

#ifdef __HLSL_VERSION
static const uint CM_SOLID = 0;
static const uint CM_FLUID = 1;
static const uint CM_AIR = 2;

// TODO: Optimize,the cell material should probably stored in R8_UINT, in a 2x2 pattern, then preloaded into shared memory this way you don't store the x,y,z neighbours and just check in smem
/*
struct SCell
{
    CellMaterial data;
};
*/
// 1) move all functions below into methods of struct above
// 2) make a special accessor or something over some provided shared memory that handles accessing neighbours from a larger texture

static const uint CellMatMask       = 0x00000003u;
static const uint CellMatMaskShift  = 0;
static const uint XPrevMatMask      = 0x0000000cu;
static const uint XPrevMatMaskShift = 2;
static const uint XNextMatMask      = 0x00000030u;
static const uint XNextMatMaskShift = 4;
static const uint YPrevMatMask      = 0x000000c0u;
static const uint YPrevMatMaskShift = 6;
static const uint YNextMatMask      = 0x00000300u;
static const uint YNextMatMaskShift = 8;
static const uint ZPrevMatMask      = 0x00000c00u;
static const uint ZPrevMatMaskShift = 10;
static const uint ZNextMatMask      = 0x00003000u;
static const uint ZNextMatMaskShift = 12;

inline void setCellMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~CellMatMask) | ((cellMaterial << CellMatMaskShift) & CellMatMask);
}
inline uint getCellMaterial(uint cellMaterials)
{
    return (cellMaterials & CellMatMask) >> CellMatMaskShift;
}
inline void setXPrevMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~XPrevMatMask) | ((cellMaterial << XPrevMatMaskShift) & XPrevMatMask);
}
inline uint getXPrevMaterial(uint cellMaterials)
{
    return (cellMaterials & XPrevMatMask) >> XPrevMatMaskShift;
}
inline void setXNextMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~XNextMatMask) | ((cellMaterial << XNextMatMaskShift) & XNextMatMask);
}
inline uint getXNextMaterial(uint cellMaterials)
{
    return (cellMaterials & XNextMatMask) >> XNextMatMaskShift;
}
inline void setYPrevMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~YPrevMatMask) | ((cellMaterial << YPrevMatMaskShift) & YPrevMatMask);
}
inline uint getYPrevMaterial(uint cellMaterials)
{
    return (cellMaterials & YPrevMatMask) >> YPrevMatMaskShift;
}
inline void setYNextMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~YNextMatMask) | ((cellMaterial << YNextMatMaskShift) & YNextMatMask);
}
inline uint getYNextMaterial(uint cellMaterials)
{
    return (cellMaterials & YNextMatMask) >> YNextMatMaskShift;
}
inline void setZPrevMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~ZPrevMatMask) | ((cellMaterial << ZPrevMatMaskShift) & ZPrevMatMask);
}
inline uint getZPrevMaterial(uint cellMaterials)
{
    return (cellMaterials & ZPrevMatMask) >> ZPrevMatMaskShift;
}
inline void setZNextMaterial(inout uint cellMaterials, uint cellMaterial)
{
    cellMaterials = (cellMaterials & ~ZNextMatMask) | ((cellMaterial << ZNextMatMaskShift) & ZNextMatMask);
}
inline uint getZNextMaterial(uint cellMaterials)
{
    return (cellMaterials & ZNextMatMask) >> ZNextMatMaskShift;
}


inline void setCellMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~CellMatMask) | ((cellMaterial << CellMatMaskShift) & CellMatMask);
}
inline uint3 getCellMaterial(uint3 cellMaterials)
{
    return (cellMaterials & CellMatMask) >> CellMatMaskShift;
}
inline void setXPrevMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~XPrevMatMask) | ((cellMaterial << XPrevMatMaskShift) & XPrevMatMask);
}
inline uint3 getXPrevMaterial(uint3 cellMaterials)
{
    return (cellMaterials & XPrevMatMask) >> XPrevMatMaskShift;
}
inline void setXNextMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~XNextMatMask) | ((cellMaterial << XNextMatMaskShift) & XNextMatMask);
}
inline uint3 getXNextMaterial(uint3 cellMaterials)
{
    return (cellMaterials & XNextMatMask) >> XNextMatMaskShift;
}
inline void setYPrevMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~YPrevMatMask) | ((cellMaterial << YPrevMatMaskShift) & YPrevMatMask);
}
inline uint3 getYPrevMaterial(uint3 cellMaterials)
{
    return (cellMaterials & YPrevMatMask) >> YPrevMatMaskShift;
}
inline void setYNextMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~YNextMatMask) | ((cellMaterial << YNextMatMaskShift) & YNextMatMask);
}
inline uint3 getYNextMaterial(uint3 cellMaterials)
{
    return (cellMaterials & YNextMatMask) >> YNextMatMaskShift;
}
inline void setZPrevMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~ZPrevMatMask) | ((cellMaterial << ZPrevMatMaskShift) & ZPrevMatMask);
}
inline uint3 getZPrevMaterial(uint3 cellMaterials)
{
    return (cellMaterials & ZPrevMatMask) >> ZPrevMatMaskShift;
}
inline void setZNextMaterial(inout uint3 cellMaterials, uint3 cellMaterial)
{
    cellMaterials = (cellMaterials & ~ZNextMatMask) | ((cellMaterial << ZNextMatMaskShift) & ZNextMatMask);
}
inline uint3 getZNextMaterial(uint3 cellMaterials)
{
    return (cellMaterials & ZNextMatMask) >> ZNextMatMaskShift;
}


inline bool isSolidCell(uint cellMaterial)
{
    return cellMaterial == CM_SOLID;
}
inline bool isFluidCell(uint cellMaterial)
{
    return cellMaterial == CM_FLUID;
}
inline bool isAirCell(uint cellMaterial)
{
    return cellMaterial == CM_AIR;
}

inline bool3 isSolidCell(uint3 cellMaterial)
{
    return cellMaterial == (uint3)CM_SOLID;
}
inline bool3 isFluidCell(uint3 cellMaterial)
{
    return cellMaterial == (uint3)CM_FLUID;
}
inline bool3 isAirCell(uint3 cellMaterial)
{
    return cellMaterial == (uint3)CM_AIR;
}

void enforceBoundaryCondition(inout float3 velocity, uint cellMaterial)
{
    bool3 is_solid_cell =
        or((bool3)isSolidCell(getCellMaterial(cellMaterial)),
        bool3(isSolidCell(getXPrevMaterial(cellMaterial)), isSolidCell(getYPrevMaterial(cellMaterial)), isSolidCell(getZPrevMaterial(cellMaterial))));
    velocity = select(is_solid_cell, 0.0f, velocity);
}

// handling solid obstacles
inline bool3 isSolidCell(float3 position)
{
    // no obstacles for now, in cuboid sim area
    return false;
}

#endif
#endif