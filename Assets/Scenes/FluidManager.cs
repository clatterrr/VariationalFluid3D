using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class FluidManager : MonoBehaviour
{
    int ParticleNum = 32 ;
    const int Nx = 8;
    int GridNum = Nx * Nx * Nx;
    int VelocityNum = (Nx + 1) * Nx * Nx;
    float dt;
    const float dx = 1.0f / Nx;
    // Start is called before the first frame update
    public Material material;

    int stride;
    int particleWarpCount;
    int gridWarpCount;
    int velWarpCount;
    int kernelIndex;
    const int WARP_SIZE = 1024;

    ComputeBuffer ParticlesPos;
    ComputeBuffer gridVelocity;
    ComputeBuffer gridVelocityTemp;
    ComputeBuffer gridVelocityTemp2;
    ComputeBuffer gridVelocityWeights;

    ComputeBuffer ACenter;
    ComputeBuffer ADown;
    ComputeBuffer AUp;
    ComputeBuffer SolidPhi;
    ComputeBuffer rhs;
    ComputeBuffer pressure;

    public ComputeShader InitRegionShader;
    public ComputeShader ParticleAdvectionShader;
    public ComputeShader GridAdvection;
    public ComputeShader GridAdvection2;
    public ComputeShader Assemble;
    public ComputeShader computeWeights;
    public ComputeShader bodyForceShader;
    public ComputeShader Substract;

    public ComputeShader solver_ScaleAdd;
    public ComputeShader solver_ScaleDot;
    public ComputeShader solver_Reduction;
    public ComputeShader solver_computeAx;

    Vector3[] initBuffer;
    Vector3[] ZeroVelGridArray;
    float[] ZeroFloatGridArray;

    PoissonManager solver = new PoissonManager();

    void Start()
    {
        particleWarpCount = Mathf.CeilToInt((float)ParticleNum / WARP_SIZE);
        gridWarpCount = Mathf.CeilToInt((float)GridNum / WARP_SIZE);
        velWarpCount = Mathf.CeilToInt((float)VelocityNum / WARP_SIZE);

        stride = Marshal.SizeOf(typeof(Vector3));

        ParticlesPos = new ComputeBuffer(ParticleNum, stride);
        gridVelocity = new ComputeBuffer(VelocityNum, stride);
        gridVelocityTemp = new ComputeBuffer(VelocityNum, stride);
        gridVelocityTemp2 = new ComputeBuffer(VelocityNum, stride);
        gridVelocityWeights = new ComputeBuffer(VelocityNum, stride);
        ADown = new ComputeBuffer(GridNum, stride);
        AUp = new ComputeBuffer(GridNum, stride);

        stride = Marshal.SizeOf(typeof(float));
        ACenter = new ComputeBuffer(GridNum, stride);
        SolidPhi = new ComputeBuffer(GridNum, stride);
        rhs = new ComputeBuffer(GridNum, stride);
        pressure = new ComputeBuffer(GridNum, stride);

        initBuffer = new Vector3[ParticleNum];
        ZeroVelGridArray = new Vector3[VelocityNum];
        ZeroFloatGridArray = new float[GridNum];

        for(int i = 0;i < GridNum;i++)
        {
            ZeroFloatGridArray[i] = 0;
        }
        SolidPhi.SetData(ZeroFloatGridArray);

        for (int i = 0; i < VelocityNum; i++)
        {
            ZeroVelGridArray[i] = new Vector3(0.0f, 0.0f, 0.0f);
        }

        for (int i = 0; i < ParticleNum; i++)
        {
            initBuffer[i] = new Vector3(i * 0.1f, 0.0f, 0.0f);
        }

        gridVelocity.SetData(ZeroVelGridArray);
        ParticlesPos.SetData(initBuffer);

        kernelIndex = InitRegionShader.FindKernel("CSMain");
        InitRegionShader.SetInt("Nx", Nx);
        InitRegionShader.SetFloat("dx", dx);
        InitRegionShader.SetBuffer(kernelIndex, "phi", SolidPhi);
        InitRegionShader.Dispatch(kernelIndex, gridWarpCount, 1, 1);

        solver.ScaleAdd = solver_ScaleAdd;
        solver.ScaleDot = solver_ScaleDot;
        solver.Reduction = solver_Reduction;
        solver.computeAx = solver_computeAx;

        material.SetBuffer("Particles", ParticlesPos);

        Step();
    }

    private void Step()
    {
        kernelIndex = ParticleAdvectionShader.FindKernel("CSMain");
        ParticleAdvectionShader.SetInt("Nx", Nx);
        ParticleAdvectionShader.SetFloat("dx", dx);
        ParticleAdvectionShader.SetFloat("dt", dt);
        ParticleAdvectionShader.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        ParticleAdvectionShader.SetBuffer(kernelIndex, "ParticlePos", ParticlesPos);
        ParticleAdvectionShader.Dispatch(kernelIndex, particleWarpCount, 1, 1);

        kernelIndex = GridAdvection.FindKernel("CSMain");
        GridAdvection.SetInt("Nx", Nx);
        GridAdvection.SetFloat("dx", dx);
        GridAdvection.SetFloat("dt", dt);
        GridAdvection.SetBuffer(kernelIndex, "old_field", gridVelocity);
        GridAdvection.SetBuffer(kernelIndex, "new_field", gridVelocityTemp);
        GridAdvection.SetBuffer(kernelIndex, "oldold_field", gridVelocityTemp2);
        GridAdvection.Dispatch(kernelIndex, velWarpCount, 1, 1);


        kernelIndex = GridAdvection2.FindKernel("CSMain");
        GridAdvection2.SetInt("Nx", Nx);
        GridAdvection2.SetFloat("dx", dx);
        GridAdvection2.SetFloat("dt", dt);
        GridAdvection2.SetBuffer(kernelIndex, "oldold_field", gridVelocityTemp2);
        GridAdvection2.SetBuffer(kernelIndex, "old_field", gridVelocityTemp);
        GridAdvection2.SetBuffer(kernelIndex, "new_field", gridVelocity);
        GridAdvection2.Dispatch(kernelIndex, velWarpCount, 1, 1);

        kernelIndex = bodyForceShader.FindKernel("CSMain");
        bodyForceShader.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        bodyForceShader.Dispatch(kernelIndex, velWarpCount, 1, 1);

        gridVelocity.GetData(ZeroVelGridArray);
        Debug.Log("wa" + ZeroVelGridArray[0]);

        kernelIndex = computeWeights.FindKernel("CSMain");
        computeWeights.SetInt("Nx", Nx);
        computeWeights.SetBuffer(kernelIndex, "VelocityWeights", gridVelocityWeights);
        computeWeights.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        computeWeights.SetBuffer(kernelIndex, "SolidPhi", SolidPhi);
        computeWeights.Dispatch(kernelIndex, velWarpCount, 1, 1);

        // 组装矩阵
        kernelIndex = Assemble.FindKernel("CSMain");
        Assemble.SetInt("Nx", Nx);
        Assemble.SetFloat("dx", dx);
        Assemble.SetFloat("dt", dt);
        Assemble.SetBuffer(kernelIndex, "AmatCenter", ACenter);
        Assemble.SetBuffer(kernelIndex, "AmatUp", AUp);
        Assemble.SetBuffer(kernelIndex, "AmatDown", ADown);
        Assemble.SetBuffer(kernelIndex, "rhs", rhs);
        Assemble.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        Assemble.SetBuffer(kernelIndex, "VelocityWeights", gridVelocityWeights);
        Assemble.Dispatch(kernelIndex, gridWarpCount, 1, 1);

        ACenter.GetData(ZeroFloatGridArray);
        for(int k = 0;k < GridNum;k++)
        {
            if(ZeroFloatGridArray[k] != 0)
            {
                Debug.Log("k = " + k + " = " + ZeroFloatGridArray[k]);
            }
        }

        // 解方程
        solver.AmatCenter = ACenter;
        solver.AmatDown = ADown;
        solver.AmatUp = AUp;
        solver.bVec = rhs;
        solver.InitBuffer();
        solver.Solve();
        pressure = solver.xVec;

        // 减去压力梯度，保证速度不为零
        kernelIndex = Substract.FindKernel("CSMain");
        Substract.SetInt("Nx", Nx);
        Substract.SetFloat("dx", dx);
        Substract.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        Substract.SetBuffer(kernelIndex, "Pressure", pressure);
        Substract.Dispatch(kernelIndex, gridWarpCount, 1, 1);
    }
    private void Update()
    {
       

    }
    void OnRenderObject()
    {
        material.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Points, 1, ParticleNum);
    }
}
