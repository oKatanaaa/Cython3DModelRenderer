import numpy as np
import cv2


class Model:
    @staticmethod
    def read_model(filename: str, silent=True, external_texture_filename=None,
                   recalculate_normals=True, invert_calculated_normals=False):

        vertices = []
        texture_coords = []
        normals = []

        triangles_vertices = []
        triangles_texture_coords = []
        triangles_normals = []

        texture = Model._read_texture_file(external_texture_filename) if external_texture_filename is not None else None

        with open(filename.strip(), 'r') as f:
            line_index = 0
            for line in f:
                try:
                    if line == '':
                        # empty line
                        continue
                    if line[0] == '#':
                        # comment line
                        continue

                    com_plus_data = line.split(' ', 1)
                    if len(com_plus_data) != 2:
                        # invalid line
                        continue

                    command, data = com_plus_data

                    if command == 'v':
                        vertices.append(Model._read_vertex(data))
                    elif command == 'vt':
                        texture_coords.append(Model._read_texture_coord(data))
                    elif command == 'vn':
                        normals.append(Model._read_normal(data))
                    elif command == 'f':
                        triangles_vs, triangles_vts, triangles_vns = Model._read_face(data)

                        triangles_vertices.extend(triangles_vs)

                        if triangles_vts.count(None) > 0:
                            triangles_texture_coords = None
                        if triangles_texture_coords is not None:
                            triangles_texture_coords.extend(triangles_vts)

                        if triangles_vns.count(None) > 0:
                            triangles_normals = None
                        if triangles_normals is not None:
                            triangles_normals.extend(triangles_vns)

                    elif command == 'mtllib' and texture is None:
                        data = (Model._get_dir(filename) if data[0] != '/' else '') + data
                        image_filename = Model._read_material_file(data, filename.strip())
                        texture = None
                        if image_filename is not None:
                            image_filename = (Model._get_dir(filename) if image_filename[0] != '/'
                                              else '') + image_filename
                            texture = Model._read_texture_file(image_filename)

                    line_index += 1

                except Exception as e:
                    if not silent:
                        raise RuntimeError(
                            f'Error occurred while parsing line #{line_index + 1} of "{filename}"') from e

        return Model(vertices, triangles_vertices,
                     texture_coords, triangles_texture_coords, texture,
                     normals, triangles_normals, recalculate_normals, invert_calculated_normals)

    @staticmethod
    def _read_material_file(filename, origin) -> str:

        image_filename = None

        try:
            with open(filename.strip(), 'r') as f:
                line_index = 0
                for line in f:
                    if line == '':
                        # empty line
                        continue
                    if line[0] == '#':
                        # comment line
                        continue

                    com_plus_data = line.split(' ', 1)
                    if len(com_plus_data) != 2:
                        # invalid line
                        continue

                    command, data = com_plus_data

                    if command == 'map_Kd':
                        image_filename = data

                    line_index += 1

        except Exception as e:
            print(f"Error occurred while parsing material file of object file '{origin}':")
            print(e)
            print('Material info will be ignored')

        return image_filename

    @staticmethod
    def _read_texture_file(filename):
        return cv2.imread(filename.strip())

    def __init__(self, vertices, triangles_vertices,
                 texture_coords=None, triangles_texture_coords=None, texture=None,
                 normals=None, triangles_normals=None, recalculate_normals=True, invert_calculated_normals=False):

        array_vertices = np.array(vertices, dtype=np.float32)
        array_triangles_vertices = np.array(triangles_vertices, dtype=np.int32)
        if normals is not None and triangles_normals is not None:
            array_normals = np.array(normals, dtype=np.float32)
            array_triangles_normals = np.array(triangles_normals, dtype=np.int32)
        else:
            array_normals = None
            array_triangles_normals = None

        self._update_vertices_and_normals(array_vertices, array_triangles_vertices,
                                          array_normals, array_triangles_normals, recalculate_normals,
                                          invert_calculated_normals)

        if texture_coords is None or triangles_texture_coords is None or texture is None:
            self._texture_coords = None
            self._triangles_texture_coords = None
            self._texture = None

            self._colors = None
            self._colors_by_triangles = None
        else:
            self._texture_coords = np.array(texture_coords, dtype=np.float32)
            self._triangles_texture_coords = np.array(triangles_texture_coords, dtype=np.int32)
            self._texture = np.array(texture)

            h, w, _ = self._texture.shape
            self._colors = self._texture[np.clip(((1 - self._texture_coords[:, 1]) * h).astype('int32'), 0, h - 1),
                                         np.clip((self._texture_coords[:, 0] * w).astype('int32'), 0, w - 1)]
            self._colors = self._colors.astype('float32')
            self._colors_by_triangles = self._colors[self._triangles_texture_coords]

    def _update_vertices_and_normals(self, array_vertices, array_triangles_vertices,
                                     array_normals, array_triangles_normals, recalculate_normals=True,
                                     invert_calculated_normals=False):
        self._vertices = array_vertices.astype('float32')
        self._triangles_vertices = array_triangles_vertices
        self._vertices_by_triangles = self._vertices[self._triangles_vertices]

        self._mean_vertex = self._vertices.mean(axis=0)
        self._max_span = np.max(np.linalg.norm(self._vertices - self._mean_vertex, axis=-1))

        if array_normals is not None and array_triangles_normals is not None and not recalculate_normals:
            self._normals = array_normals.astype('float32')
            self._triangles_normals = array_triangles_normals
        else:
            self._normals = Model._compute_normals_by_vertex(self._vertices, self._triangles_vertices)
            self._triangles_normals = self._triangles_vertices
            if invert_calculated_normals:
                self._normals *= -1

        self._normals_by_triangles = self._normals[self._triangles_normals]

    @staticmethod
    def _compute_normals_by_vertex(vertices, triangles_vertices, duplicate_normal_dot_tolerance=0):
        # Vertex normals are meant
        all_normals_of_vertex = [[] for _ in range(len(vertices))]
        for triangle_vertices in triangles_vertices:
            n = Model._compute_triangle_normal(vertices[triangle_vertices], normalize=True)
            for vertex_index in triangle_vertices:
                new = True
                for existing_normal in all_normals_of_vertex[vertex_index]:
                    if np.dot(existing_normal, n) >= 1 - duplicate_normal_dot_tolerance:
                        new = False
                if new:
                    all_normals_of_vertex[vertex_index].append(n)
        return np.stack([Model._normalize(np.mean(np.stack(normals), axis=0)) if len(normals) > 0
                         else np.zeros(shape=3, dtype=np.float32) for normals in all_normals_of_vertex]).astype('float32')

    @staticmethod
    def _normalize(n):
        if np.linalg.norm(n) == 0:
            return n
        return n / np.linalg.norm(n)

    @staticmethod
    def _compute_triangle_normal(triangle, normalize=True):
        n = -np.cross(triangle[1]-triangle[0], triangle[1]-triangle[2])
        if normalize:
            n = Model._normalize(n)
        return n

    def get_vertex(self, index: int):
        return self._vertices[index], \
               (self._colors[index] if self._colors is not None else None), \
               self._normals[index]

    def get_triangle(self, index: int):
        return self._vertices_by_triangles[index], \
               (self._colors_by_triangles[index] if self._colors_by_triangles is not None else None), \
               self._normals_by_triangles[index]

    def shift(self, shift):
        new_vertices = self._vertices + shift
        self._update_vertices_and_normals(new_vertices, self._triangles_vertices,
                                          self._normals, self._triangles_normals, recalculate_normals=False)

    def scale(self, scale_coef, keep_position=True):
        new_vertices = self._vertices
        if keep_position:
            new_vertices -= self._mean_vertex
            new_vertices *= scale_coef
            new_vertices += self._mean_vertex
        else:
            new_vertices *= scale_coef
        self._update_vertices_and_normals(new_vertices, self._triangles_vertices,
                                          self._normals, self._triangles_normals, recalculate_normals=False)

    @staticmethod
    def _rot_matrix(angle, degrees=True):
        if degrees:
            angle *= np.pi / 180
        return np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])

    def rotate(self, angles):
        assert len(angles) == 3

        angle_x, angle_y, angle_z = angles

        mat_rot_x = np.eye(3)
        mat_rot_x[1:, 1:] = Model._rot_matrix(angle_x)

        mat_rot_y = np.eye(3)
        mat_rot_y[::2, ::2] = Model._rot_matrix(angle_y)

        mat_rot_z = np.eye(3)
        mat_rot_z[:2, :2] = Model._rot_matrix(angle_z)

        mat_rot = np.matmul(np.matmul(mat_rot_x, mat_rot_y), mat_rot_z)

        new_vertices = np.matmul(self._vertices, np.transpose(mat_rot))

        self._update_vertices_and_normals(new_vertices, self._triangles_vertices, None, None, recalculate_normals=True)

    def n_triangles(self) -> int:
        return len(self._triangles_vertices)

    def n_vertices(self) -> int:
        return len(self._vertices)

    @staticmethod
    def _read_vertex(data: str):
        coords = [float(t) for t in data.split()]
        assert len(coords) >= 3
        return coords[:3]  # reading X Y Z ignoring optional W coordinate

    @staticmethod
    def _read_texture_coord(data: str):
        return [float(t) for t in data.split()]  # reading all U V W including maybe optional V W

    @staticmethod
    def _read_normal(data: str):
        coords = [float(t) for t in data.split()]
        assert len(coords) == 3
        return coords  # reading X Y Z explicitly

    @staticmethod
    def _fix_index(index: int) -> int:
        if index > 0:
            index -= 1
        return index

    @staticmethod
    def _read_face(data: str):
        comp = data.split()
        triangles = [[comp[0], comp[1 + i], comp[2 + i]] for i in range(len(comp) - 2)]  # triangulation
        triangles_vs = []
        triangles_vts = []
        triangles_vns = []
        for triangle in triangles:
            triangle_vs = []
            triangle_vts = []
            triangle_vns = []
            for comp in triangle:
                v, vt, vn = (comp + '//').split('/')[:3]

                triangle_vs.append(Model._fix_index(int(v)))

                if vt == '':
                    triangle_vts = None
                if triangle_vts is not None:
                    triangle_vts.append(Model._fix_index(int(vt)))

                if vn == '':
                    triangle_vns = None
                if triangle_vns is not None:
                    triangle_vns.append(Model._fix_index(int(vn)))

            triangles_vs.append(triangle_vs)
            triangles_vts.append(triangle_vts)
            triangles_vns.append(triangle_vns)

        return triangles_vs, triangles_vts, triangles_vns

    def get_mean_vertex(self):
        return self._mean_vertex

    def get_max_span(self):
        return self._max_span

    @staticmethod
    def _get_dir(filename):
        li = filename.rsplit('/', 1)
        if len(li) < 2:
            return ''
        return li[-2] + '/'
