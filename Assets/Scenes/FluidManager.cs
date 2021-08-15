using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class FluidManager : MonoBehaviour
{
    int ParticleNum = 512;
    const int Nx = 16;
    int GridNum = Nx * Nx * Nx;
    // 16 **3 = 4096
    int VelocityNum = (Nx + 1) * Nx * Nx;
    float dt = 0.05f;
    const float dx = 1.0f / (float)Nx;
    // Start is called before the first frame update
    public Material material;

    int stride;
    int particleWarpCount;
    int gridWarpCount;
    int velWarpCount;
    int kernelIndex;
    const int WARP_SIZE = 512;

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

    public GameObject spherePrefab;
    GameObject[] sphere;
    Vector3[] DataParticle3D;
    Vector3[] DataVel3D;
    Vector3[] DataGrid3d;
    float[] DataGrid;

    PoissonManager solver = new PoissonManager();

    void Start()
    {
        sphere = new GameObject[ParticleNum];

        particleWarpCount = Mathf.CeilToInt((float)ParticleNum / WARP_SIZE) ;
        gridWarpCount = Mathf.CeilToInt((float)GridNum / WARP_SIZE) ;
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

        DataParticle3D = new Vector3[ParticleNum];
        DataVel3D = new Vector3[VelocityNum];
        DataGrid = new float[GridNum];

        for (int i = 0; i < GridNum; i++)
        {
            DataGrid[i] = 0;
        }
        SolidPhi.SetData(DataGrid);

        for (int i = 0; i < VelocityNum; i++)
        {
            DataVel3D[i] = new Vector3(0.0f, 0.0f, 0.0f);

        }

        for (int i = 0; i < ParticleNum; i++)
        {
            float px = Random.Range(0.4f, 0.6f);
            float py = Random.Range(0.6f, 0.8f);
            float pz = Random.Range(0.4f, 0.6f);
            pz = 0.5f;
            DataParticle3D[i] = new Vector3(px, py, pz);
            sphere[i] = Instantiate(spherePrefab, DataParticle3D[i], Quaternion.identity);
        }

        gridVelocity.SetData(DataVel3D);
        ParticlesPos.SetData(DataParticle3D);

        kernelIndex = InitRegionShader.FindKernel("CSMain");
        InitRegionShader.SetInt("Nx", Nx);
        InitRegionShader.SetFloat("dx", dx);
        InitRegionShader.SetBuffer(kernelIndex, "phi", SolidPhi);
        InitRegionShader.SetBuffer(kernelIndex, "pos", ParticlesPos);
        InitRegionShader.Dispatch(kernelIndex, gridWarpCount, 1, 1);

        solver.ScaleAdd = solver_ScaleAdd;
        solver.ScaleDot = solver_ScaleDot;
        solver.Reduction = solver_Reduction;
        solver.computeAx = solver_computeAx;

        

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
        bodyForceShader.SetInt("Nx", Nx);
        bodyForceShader.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        bodyForceShader.Dispatch(kernelIndex, velWarpCount, 1, 1);

        gridVelocity.GetData(DataVel3D);
        //Debug.Log("wa" + DataVel3D[Nx * Nx * Nx / 2 + Nx * Nx / 2 + Nx / 2].ToString("f4"));

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

        gridVelocity.GetData(DataVel3D);
       // Debug.Log(DataVel3D[Nx * Nx * Nx / 2 + Nx * Nx / 2 + Nx / 2]);
        for (int i = 0; i < VelocityNum; i++)
        {
            //if (DataVel3D[i].x != 0) Debug.Log("i =" + i + " = " + DataVel3D[i]);
        }

        rhs.GetData(DataGrid);
        for (int i = 0; i < GridNum; i++)
        {
            // if (DataGrid[i] != 0) Debug.Log("i =" + i + " = " + DataGrid[i]);
        }

        // 解方程
        solver.AmatCenter = ACenter;
        solver.AmatDown = ADown;
        solver.AmatUp = AUp;
        solver.bVec = rhs;
        solver.Nx = Nx;
        solver.InitBuffer();
        solver.Solve();
        pressure = solver.xVec;


        // 减去压力梯度，保证速度不为零
        kernelIndex = Substract.FindKernel("CSMain");
        Substract.SetInt("Nx", Nx);
        Substract.SetFloat("dx", dx);
        Substract.SetFloat("dt", dt);
        Substract.SetBuffer(kernelIndex, "Velocity", gridVelocity);
        Substract.SetBuffer(kernelIndex, "Pressure", pressure);
        Substract.Dispatch(kernelIndex, gridWarpCount, 1, 1);
    }
    int cnt = 0;
    private void Update()
    {
        cnt += 1;
       // if (cnt > 10) return;
        Step();
        ParticlesPos.GetData(DataParticle3D);
        for (int i = 0; i < ParticleNum; i++)
        {
            sphere[i].transform.position = DataParticle3D[i];
            //Debug.Log("pos = " + DataParticle3D[i]);
        }
        gridVelocity.GetData(DataVel3D);
        for (int i = 0; i < VelocityNum; i++)
        {
            int basex = i % (Nx + 1);
            int basey = i % ((Nx + 1) * Nx) / (Nx + 1);
            int basez = i / ((Nx + 1) * Nx);

            //Debug.Log("x = " + basex + " y =  " + basey + " z = " + basez + " so vel = " + DataVel3D[i].x.ToString("f4"));

            basex = i % Nx;
            basey = i % ((Nx + 1) * Nx) / Nx;
            basez = i / ((Nx + 1) * Nx);

            //Debug.Log("x = " + basex + " y =  " + basey + " z = " + basez + " so vel = " + DataVel3D[i].y.ToString("f4"));

            basex = i % Nx;
            basey = i % (Nx  * Nx) / Nx;
            basez = i / (Nx * Nx);

            //Debug.Log("x = " + basex + " y =  " + basey + " z = " + basez + " so vel = " + DataVel3D[i].z.ToString("f4"));
        }

    }
}
