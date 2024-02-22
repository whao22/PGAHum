import trimesh
mesh = trimesh.load("test/00050000.ply")
vertices = mesh.vertices

print(mesh)