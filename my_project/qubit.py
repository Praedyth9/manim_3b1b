from manimlib.imports import *
from my_project.qubit_utils import *
import math as maths
import sympy as sp
###For inverse trig calc, esp. for mp.atan2(y,x) calculation
from mpmath import mp
import numpy as np
import quaternionic

OUTPUT_DIRECTORY = "qubit"


class State(Mobject):
	def __init__(self, zero_amplitude, one_amplitude, r=SPHERE_RADIUS, **kwargs):
		Mobject.__init__(self, **kwargs)

		self.zero_amplitude = complex(zero_amplitude)
		self.one_amplitude = complex(one_amplitude)

		self.r = r
		self.theta, self.phi = vector_to_angles(self.get_vector())

		self.line = self.create_line()
		self.add(self.line)

	def spherical_to_cartesian(r,theta,phi):
		return [r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi),r*math.cos(theta)]
	
	def _get_cartesian(self):
		return np.array( spherical_to_cartesian(self.r, self.theta, self.phi) )

	def create_line(self):
		return Line(
			start=ORIGIN,
			end=self._get_cartesian(),
		)

	def get_vector(self):
		return np.array([self.zero_amplitude, self.one_amplitude])

	def apply_operator(self, operator, verbose=True):
		if verbose:
			print("from: ", self.get_vector())
		vector_result = operator.dot(self.get_vector())
		if verbose:
			print("to  : ", vector_result)
		new_state = State(*vector_result)
		new_state.set_color(self.color)
		return new_state
	

state_zero  = State(1,             0,            r=SPHERE_RADIUS)
state_one   = State(0,             1,            r=SPHERE_RADIUS)
state_plus  = State(1/np.sqrt(2),  1/np.sqrt(2), r=SPHERE_RADIUS)
state_minus = State(1/np.sqrt(2), -1/np.sqrt(2), r=SPHERE_RADIUS)


class Matrix(Mobject):
	def construct(self):
		m0 = Matrix([[1, 0], [0, 1]])
		self.add(m0)
	
	def apply_operator_matrix(self,operator, verbose = True):
		if verbose:
			print("from: ", self.m0)
		result = operator
		if verbose:
			print("to : ", result)
		new_matrix = Matrix(operator.dot(self.m0))
		new_matrix.set_color(WHITE)
		return new_matrix
		



class BlochSphere(SpecialThreeDScene):
	CONFIG = {
		"three_d_axes_config": {
			"num_axis_pieces": 1,
			"number_line_config": {
				"unit_size": 2,
				"tick_frequency": 1,
				"numbers_with_elongated_ticks": [0, 1, 2],
				"stroke_width": 2,
			}
		},
		"init_camera_orientation": {
			"phi": 80 * DEGREES,
			# "theta": -135 * DEGREES,
			"theta": 15 * DEGREES,
		},

		"circle_xz_show": False,
		"circle_xz_color": PINK,

		"circle_xy_show": True,
		"circle_xy_color": GREEN,

		"circle_yz_show": False,
		"circle_yz_color": ORANGE,

		
		"sphere_config": {
			"radius": SPHERE_RADIUS,
			"resolution": (60, 60),
		},
		
		"rotate_sphere": True,
		"rotate_circles": False,
		"rotate_time": 5,
		"operators": [
		],
		"operator_names": [
		],
		"show_intro": True,

		"wait_time": 2,
		"pre_operators_wait_time": 1.5,
		"final_wait_time": 3,
		"intro_wait_time": 3,
		"intro_fadeout_wait_time": 1,
	}

	def construct(self):
		if self.show_intro:
			self.present_introduction()
		self.init_camera()
		self.init_axes()
		self.init_sphere()
		self.init_states()
		self.init_text()
		self.wait(self.pre_operators_wait_time)

		for o in self.operators:
			self.apply_operator(o)
			#self.apply_operator_matrix(o)
			self.wait(self.wait_time)
		self.wait(self.final_wait_time)

	
	def present_introduction(self):
		self.intro_tex_1 = TextMobject("Représentation de deux qubits")
		self.intro_tex_1.move_to(2*UP + 2.5*LEFT)

		self.intro_tex_12 = TextMobject("$\\ket{0}$", color = BLUE) 
		self.intro_tex_12.next_to(self.intro_tex_1, RIGHT)

		self.intro_tex_13 = TextMobject(" et ")
		self.intro_tex_13.next_to(self.intro_tex_12, RIGHT)

		self.intro_tex_14 = TextMobject("$\\ket{1}$", color = RED)
		self.intro_tex_14.next_to(self.intro_tex_13, RIGHT)

		self.intro_tex_15 = TextMobject("non intriqués")
		self.intro_tex_15.next_to(self.intro_tex_14, RIGHT)

		self.intro_tex_16 = TextMobject("sur la sphère de Bloch.")
		self.intro_tex_16.move_to(1*UP)

		print(self.intro_tex_1)
		#self.intro_tex_1[2].set_color(BLUE)
		#self.intro_tex_1[4].set_color(color=RED)
		#self.intro_tex_1[2].set_color_by_tex('$\\ket{0}$', BLUE)
		#self.intro_tex_1[4].set_color_by_tex('$\\ket{1}$', RED)
		
		#self.intro_tex_1 = TextMobject(
			#"\\begin{flushleft}\n"
			#"The State of the Qbit"
			#"\\\\"
			#"as represented in the Bloch Sphere."
			#"\n\\end{flushleft}"
		#)
		# self.intro_tex_1 = TextMobject(
		# 	# "\\begin{align*}\n" + "The state of the Qbit" + "\n\\end{align*}",
		# 	"\\begin{flalign}\n" + "The state of the Qbit" + "\n\\end{flalign}",
		# 	# "The state of the Qbit",
		# 	# "\\begin{flushleft}"
		# 	# "The state of the Qbit"
		# 	# "\\\\"
		# 	# "as represented in the Bloch Sphere."
		# 	# "\\end{flushleft}",
		# 	alignment="",
		# 	# template_tex_file_body=TEMPLATE_TEXT_FILE_BODY,
	 #        # arg_separator="",
		# )
		self.add(self.intro_tex_1)
		self.wait(0.5)
		self.play(
			Write(self.intro_tex_1), Write(self.intro_tex_12), Write(self.intro_tex_13), Write(self.intro_tex_14), Write(self.intro_tex_15), Write(self.intro_tex_16),
			run_time=1.5
		)
		self.wait(0.5)
		
		if self.operator_names:
			self.intro_tex_2 = TextMobject(
				"\\begin{flushleft}"
				#"The following gates will be applied:"
				"Nous appliquons sur ces qubits les portes suivantes :"
				"\\\\"
				+
				"\\\\".join(f"{i+1}) {n}" for i,n in enumerate(self.operator_names))
				+
				"\n\\end{flushleft}"
			)
			self.intro_tex_2.move_to(0.8*DOWN)
			self.add(self.intro_tex_2)
			self.play(
				Write(self.intro_tex_2),
				run_time=2.5
			)
		
		#self.intro_tex_1 = tex("Application de la matrice H : $\frac{1}{2} \begin{pmatrix}\n 1 & 1\\ \n 1 & -1 \n \end{pmatrix}$")
		self.wait(self.intro_wait_time)

		if self.operator_names:
			self.play(
				FadeOut(self.intro_tex_1),
				FadeOut(self.intro_tex_12),
				FadeOut(self.intro_tex_13),
				FadeOut(self.intro_tex_14),
				FadeOut(self.intro_tex_15),
				FadeOut(self.intro_tex_16),
				FadeOut(self.intro_tex_2)
			)
		else:
			self.play(
				FadeOut(self.intro_tex_1),
				FadeOut(self.intro_tex_12),
				FadeOut(self.intro_tex_13),
				FadeOut(self.intro_tex_14),
				FadeOut(self.intro_tex_15),
				FadeOut(self.intro_tex_16)
			)

		self.wait(self.intro_fadeout_wait_time)
	
	def init_camera(self):
		self.set_camera_orientation(**self.init_camera_orientation)

	def init_axes(self):
		self.axes = self.get_axes()
		self.set_axes_labels()
		self.add(self.axes)

	def _tex(self, *s):
		tex = TexMobject(*s)
		tex.rotate(90 * DEGREES, RIGHT)
		tex.rotate(90 * DEGREES, OUT)
		tex.scale(0.5)
		return tex

	def set_axes_labels(self):
		labels = VGroup()

		zero = tex("\\ket{0}")
		zero.next_to(
			self.axes.z_axis.number_to_point(1),
			Y_AXIS + Z_AXIS,
			MED_SMALL_BUFF
		)

		one = tex("\\ket{1}")
		one.next_to(
			self.axes.z_axis.number_to_point(-1),
			Y_AXIS - Z_AXIS,
			MED_SMALL_BUFF
		)

		labels.add(zero, one)
		self.axes.z_axis.add(labels)

		x = tex("x")
		x.next_to(
			self.axes.x_axis.number_to_point(1),
			-Y_AXIS,
			MED_SMALL_BUFF
		)
		self.axes.x_axis.add(x)

		y = tex("y")
		y.next_to(
			self.axes.y_axis.number_to_point(1),
			Y_AXIS + Z_AXIS,
			MED_SMALL_BUFF
		)
		self.axes.y_axis.add(y)

	def init_sphere(self):
		sphere = self.get_sphere(**self.sphere_config)
		sphere.set_fill(BLUE_E)
		sphere.set_opacity(0.1)
		self.add(sphere)
		self.sphere = sphere

		if self.circle_xy_show:
			self.circle_xy = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_xy_color,
			)
			self.circle_xy.set_fill(self.circle_xy_color)
			self.circle_xy.set_opacity(0.1)
			self.add(self.circle_xy)

		if self.circle_xz_show:
			self.circle_xz = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_xz_color,
			)
			self.circle_xz.rotate(90 * DEGREES, RIGHT)
			self.circle_xz.set_fill(self.circle_xz_color)
			self.circle_xz.set_opacity(0.1)
			self.add(self.circle_xz)

		if self.circle_yz_show:
			self.circle_yz = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_yz_color,
			)
			self.circle_yz.rotate(90 * DEGREES, UP)
			self.circle_yz.set_fill(self.circle_yz_color)
			self.circle_yz.set_opacity(0.1)
			self.add(self.circle_yz)

	def init_text(self):
		"""
		for each state, write (with its own color):
			the probabilities
			theta & phi
		"""
		# the qquad is used as a placeholder, since the value changes, and the length of the value changes.

		#self.tex_matrix = tex("\\frac{1}{\sqrt{2}} \\begin{bmatrix} 1 & 1 \\\ 1 & -1 \\end{bmatrix}")
		self.tex_matrix = tex("Transformation : ", "\\\\", "\\\\", "\\qquad \\begin{bmatrix} 1.000 & 0.000 \\\ 0.000 & 1.000 \end{bmatrix}")
		self.tex_matrix.set_color(WHITE)
		self.tex_matrix.move_to(- Z_AXIS * 2 + Y_AXIS * 3)

		self.tex_zero_vec   = tex("\\ket{BLUE} = ", "\\qquad \\qquad 1", " \\\\ ", "\\qquad 0")
		self.tex_zero_vec.set_color(BLUE)
		self.tex_zero_vec.move_to(Z_AXIS * 2 - Y_AXIS * 4)

		self.tex_zero_theta = tex("\\theta = ", "0.000")
		self.tex_zero_theta.set_color(BLUE)
		self.tex_zero_theta.move_to(Z_AXIS * 1 - Y_AXIS * 4)

		self.tex_zero_phi   = tex("\\phi = ", "0.000")
		self.tex_zero_phi.set_color(BLUE)
		self.tex_zero_phi.move_to(Z_AXIS * 0.5 - Y_AXIS * 4)


		self.tex_one_vec    = tex("\\ket{RED} = ", "\\qquad \\qquad 0", " \\\\ ", "\\qquad 1")
		self.tex_one_vec.set_color(RED)
		self.tex_one_vec.move_to(Z_AXIS * 2 + Y_AXIS * 3.5)

		self.tex_one_theta  = tex("\\theta = ", "180.0")
		self.tex_one_theta.set_color(RED)
		self.tex_one_theta.move_to(Z_AXIS * 1 + Y_AXIS * 4)

		self.tex_one_phi    = tex("\\phi = ", "0.000")
		self.tex_one_phi.set_color(RED)
		self.tex_one_phi.move_to(Z_AXIS * 0.5 + Y_AXIS * 4)

		self.tex_dot_product= tex("\\bra{0}\\ket{1} = ", "\\qquad \\quad 0.000")
		self.tex_dot_product.set_color(WHITE)
		self.tex_dot_product.move_to(- Z_AXIS * 2 - Y_AXIS * 5.5)

		self.add(
			self.tex_matrix,

			self.tex_zero_vec,
			self.tex_zero_theta,
			self.tex_zero_phi,

			self.tex_one_vec,
			self.tex_one_theta,
			self.tex_one_phi,

			self.tex_dot_product,
		)

		# the initial values are only used to make enough space for later values
		self.play(
			*self.update_tex_transforms(self.zero, self.one, self.matrix),
			run_time=0.1
		)

	def update_tex_transforms(self, new_zero, new_one, new_matrix):
		zero_state = new_zero.get_vector()
		zero_angles = vector_to_angles(zero_state)
		one_state = new_one.get_vector()
		one_angles = vector_to_angles(one_state)
		print(new_matrix)
		dot_product = np.vdot( new_one.get_vector(), new_zero.get_vector())

		return(
			transform(self.tex_matrix[3], matrix_to_tex_string(new_matrix)),
			transform(self.tex_zero_vec[1],   complex_to_str(zero_state[0])),
			transform(self.tex_zero_vec[3],   complex_to_str(zero_state[1])),
			transform(self.tex_zero_theta[1], angle_to_str(zero_angles[0]) ),
			transform(self.tex_zero_phi[1],   angle_to_str(zero_angles[1]) ),

			transform(self.tex_one_vec[1],    complex_to_str(one_state[0]) ),
			transform(self.tex_one_vec[3],    complex_to_str(one_state[1]) ),
			transform(self.tex_one_theta[1],  angle_to_str(one_angles[0])  ),
			transform(self.tex_one_phi[1],    angle_to_str(one_angles[1])  ),

			transform(self.tex_dot_product[1],   complex_to_str(dot_product)),
		)

	def init_states(self):
		self.old_zero = self.zero = State(1, 0, r=2)
		self.old_one  = self.one  = State(0, 1, r=2)
		self.zero.set_color(BLUE)
		self.one.set_color(RED)
		self.old_matrix = self.matrix = Matrix()
		self.add(self.zero, self.one, self.matrix)

	def apply_operator(self, operator, verbose=True):
		# preparing the rotation animation
		vg = VGroup(self.old_zero.line, self.old_one.line)
		if self.rotate_sphere:
			vg.add(self.sphere)

		if self.rotate_circles:
			if self.circle_xy_show:
				vg.add(self.circle_xy)
			if self.circle_xz_show:
				vg.add(self.circle_xz)
			if self.circle_yz_show:
				vg.add(self.circle_yz)


		rm = RotationMatrix(operator)

		if verbose:
			print(f"rotating around axis: {rm.axis} by {rm.theta / DEGREES} degrees")

		# preparing the tex update
		new_zero = self.zero.apply_operator(operator)
		new_one = self.one.apply_operator(operator)
		new_matrix = np.round(operator, 3)
		'''
		if (new_matrix == np.array([[0.707, 0.707], [0.707,-0.707]])).all():
			new_matrix = Matrix([["\frac{1}{\sqrt{2}}", "\frac{1}{\sqrt{2}}"], ["\frac{1}{\sqrt{2}}", "- \frac{1}{\sqrt{2}}"]])
		'''
		#Matrix library not working since this manim version is too old, so had to create a specific class as a placeholder but can't print latex formulas

		self.play(
			Rotate(
				vg,
				angle=rm.theta,
				axis=rm.axis
			),
			*self.update_tex_transforms(new_zero, new_one, new_matrix),
			run_time=self.rotate_time
		)

		self.zero = new_zero
		self.one  = new_one

	def apply_operator_old(self, operator, verbose=True):
		if verbose:
			print()
			print("00000")
		new_zero = self.zero.apply_operator(operator)
		if verbose:
			print("11111")
		new_one = self.one.apply_operator(operator)

		self.play(
			Transform(self.old_zero, new_zero),
			Transform(self.old_one,  new_one),
			#Transform(self.matrix, operator),
			*self.update_tex_transforms(new_zero, new_one),
		)

		self.zero = new_zero
		self.one  = new_one
		#self.matrix = operator


class BlochSphereHadamardRotate(BlochSphere):
	CONFIG = {
		"show_intro": False,
		"rotate_sphere": True,
		"rotate_time": 5,
		"rotate_amount": 1,
	}

	def construct(self):
		if self.show_intro:
			self.present_introduction()
		self.init_camera()
		self.init_axes()
		self.init_sphere()
		self.init_states()
		self.init_text()
		self.wait(self.pre_operators_wait_time)

		self.init_rotation_axis()

		for _ in range(self.rotate_amount):
			self.haramard_rotate()
			self.wait()
		self.wait(self.final_wait_time)

	def init_rotation_axis(self):
		self.direction = 1/np.sqrt(2) * (X_AXIS + Z_AXIS)

		d = Line(
			start=ORIGIN,
			end=self.direction * SPHERE_RADIUS
		)

		x_arc = Arc(
			arc_center=ORIGIN,
			start_angle=0 * DEGREES,
			angle=45 * DEGREES,
			**{
				"radius": 1,
				"stroke_color": GREEN,
				"stroke_width": 2,
			},
		)

		z_arc = Arc(
			arc_center=ORIGIN,
			start_angle=90 * DEGREES,
			angle=-45 * DEGREES,
			**{
				"radius": 0.8,
				"stroke_color": PINK,
				"stroke_width": 2,
			},
		)
		x_arc.rotate_about_origin(90 * DEGREES, X_AXIS)
		z_arc.rotate_about_origin(90 * DEGREES, X_AXIS)

		self.add(d, x_arc, z_arc)

	def haramard_rotate(self):
		a = VGroup(self.old_zero.line, self.old_one.line)
		if self.rotate_sphere:
			a.add(self.sphere)

		self.play(
			Rotate(
				a,
				angle=PI,
				axis=self.direction
			),
			run_time=self.rotate_time
		)

'''
OLD
Since one qubit can be represented in S^3, we can easily say that the tensored states of two qubits is in S^7, by again removing the global phase of state ket 00.

For more convenience for display an element of S^7, we're using Hopf fibration to be allowed to represent them on three different plots according to this article about the [Bloch Sphere for two qubits pure states](https://arxiv.org/abs/1403.8069). 

For this manim project, the representation of the [two-qubits bloch sphere](https://arxiv.org/abs/2003.01699) depends on the base qubit we do choose. For the currently displayed animation, the first qubit (A) is the base qubit. It will always be represented on the Quasi Bloch Sphere ($\mathcal{S}^2$). The second qubit (B) will be correctly be displayed if both qubits are not entangled, otherwise the qubit B will be displayed as a pure state ($\ket{0}$ or $\ket{1}$) which corresponds to the state the second qubit will be mesured if the qubit A is measured at the state $\ket{0}$.

The computation of the states of the qubits displayed is allowed by this [Github repository](https://github.com/CRWie/two-qubit-Bloch-sphere). 

TO DO : Extract baseA qubit coord thx to quaternion , find the rotation matrix and decompose it in the base of SU(2)
Focus more on pure states only, excluding cases when it's fully entrangled
When fully entrangled, Fiber B will plot state of qubit B when qubit A has 0 as state
'''

'''
NEW

2 qubits tensored correspond to a vector with 4 lines. The implementation will be in a more practical way following the following rules. The obtained result is the same as the OLD method :

- We keep the system of base qubit (= q0) and fiber qubit (= q1)

- First step is to compute the superposition of tensored states --> operator [matrix 4x4] * [vector size 4]

- Second step is to check if the two qubits are entangled

- Third step 1 : if not entangled, get angles theta and phi of q0 based on BaseACalculation.py (https://github.com/CRWie/two-qubit-Bloch-sphere), call angle_to_vector func and solve basic equations to get q1

- Third step 2 : if entangled, collect value to attribute them to q0, q1 corresponds to the state when q0 is at ket 0
'''
class BlochSphere2qubits(SpecialThreeDScene):
	CONFIG = {
		"three_d_axes_config": {
			"num_axis_pieces": 1,
			"number_line_config": {
				"unit_size": 2,
				"tick_frequency": 1,
				"numbers_with_elongated_ticks": [0, 1, 2],
				"stroke_width": 2,
			}
		},
		"init_camera_orientation": {
			"phi": 80 * DEGREES,
			# "theta": -135 * DEGREES,
			"theta": 15 * DEGREES,
		},

		"circle_xz_show": False,
		"circle_xz_color": PINK,

		"circle_xy_show": True,
		"circle_xy_color": GREEN,

		"circle_yz_show": False,
		"circle_yz_color": ORANGE,

		
		"sphere_config": {
			"radius": SPHERE_RADIUS,
			"resolution": (60, 60),
		},
		
		"rotate_sphere": True,
		"rotate_circles": False,
		"rotate_time": 5,
		"operators": [
		],
		"operator_names": [
		],
		"show_intro": True,

		"wait_time": 2,
		"pre_operators_wait_time": 1.5,
		"final_wait_time": 3,
		"intro_wait_time": 3,
		"intro_fadeout_wait_time": 1,
	}

	def construct(self):
		#if self.show_intro:
		#	self.present_introduction()
		self.init_camera()
		self.init_axes()
		self.init_sphere()
		self.init_states()
		self.init_text()
		self.wait(self.pre_operators_wait_time)

		for o in self.operators:
			self.apply_operator(o)
			#self.apply_operator_matrix(o)
			self.wait(self.wait_time)
		self.wait(self.final_wait_time)

	
	def present_introduction(self):
		self.intro_tex_1 = TextMobject("Représentation de deux qubits")
		self.intro_tex_1.move_to(2*UP + 2.5*LEFT)

		self.intro_tex_12 = TextMobject("$\\ket{q_0}$", color = BLUE) 
		self.intro_tex_12.next_to(self.intro_tex_1, RIGHT)

		self.intro_tex_13 = TextMobject(" et ")
		self.intro_tex_13.next_to(self.intro_tex_12, RIGHT)

		self.intro_tex_14 = TextMobject("$\\ket{q_1}$", color = RED)
		self.intro_tex_14.next_to(self.intro_tex_13, RIGHT)

		self.intro_tex_16 = TextMobject("sur la sphère de Bloch.")
		self.intro_tex_16.move_to(1*UP)

		print(self.intro_tex_1)
		self.add(self.intro_tex_1)
		self.wait(0.5)
		self.play(
			Write(self.intro_tex_1), Write(self.intro_tex_12), Write(self.intro_tex_13), Write(self.intro_tex_14),  Write(self.intro_tex_16),
			run_time=1.5
		)
		self.wait(0.5)
		
		if self.operator_names:
			self.intro_tex_2 = TextMobject(
				"\\begin{flushleft}"
				#"The following gates will be applied:"
				"Nous appliquons sur ces qubits les portes suivantes :"
				"\\\\"
				+
				"\\\\".join(f"{i+1}) {n}" for i,n in enumerate(self.operator_names))
				+
				"\n\\end{flushleft}"
			)
			self.intro_tex_2.move_to(0.8*DOWN)
			self.add(self.intro_tex_2)
			self.play(
				Write(self.intro_tex_2),
				run_time=2.5
			)
		
		#self.intro_tex_1 = tex("Application de la matrice H : $\frac{1}{2} \begin{pmatrix}\n 1 & 1\\ \n 1 & -1 \n \end{pmatrix}$")
		self.wait(self.intro_wait_time)

		if self.operator_names:
			self.play(
				FadeOut(self.intro_tex_1),
				FadeOut(self.intro_tex_12),
				FadeOut(self.intro_tex_13),
				FadeOut(self.intro_tex_14),
				FadeOut(self.intro_tex_16),
				FadeOut(self.intro_tex_2)
			)
		else:
			self.play(
				FadeOut(self.intro_tex_1),
				FadeOut(self.intro_tex_12),
				FadeOut(self.intro_tex_13),
				FadeOut(self.intro_tex_14),
				FadeOut(self.intro_tex_16)
			)

		self.wait(self.intro_fadeout_wait_time)
	
	def init_camera(self):
		self.set_camera_orientation(**self.init_camera_orientation)

	def init_axes(self):
		self.axes = self.get_axes()
		self.set_axes_labels()
		self.add(self.axes)

	def _tex(self, *s):
		tex = TexMobject(*s)
		tex.rotate(90 * DEGREES, RIGHT)
		tex.rotate(90 * DEGREES, OUT)
		tex.scale(0.5)
		return tex

	def set_axes_labels(self):
		labels = VGroup()

		zero = tex("\\ket{0}")
		zero.next_to(
			self.axes.z_axis.number_to_point(1),
			Y_AXIS + Z_AXIS,
			MED_SMALL_BUFF
		)

		one = tex("\\ket{1}")
		one.next_to(
			self.axes.z_axis.number_to_point(-1),
			Y_AXIS - Z_AXIS,
			MED_SMALL_BUFF
		)

		labels.add(zero, one)
		self.axes.z_axis.add(labels)

		x = tex("x")
		x.next_to(
			self.axes.x_axis.number_to_point(1),
			-Y_AXIS,
			MED_SMALL_BUFF
		)
		self.axes.x_axis.add(x)

		y = tex("y")
		y.next_to(
			self.axes.y_axis.number_to_point(1),
			Y_AXIS + Z_AXIS,
			MED_SMALL_BUFF
		)
		self.axes.y_axis.add(y)

	def init_sphere(self):
		sphere = self.get_sphere(**self.sphere_config)
		sphere.set_fill(BLUE_E)
		sphere.set_opacity(0.1)
		self.add(sphere)
		self.sphere = sphere

		if self.circle_xy_show:
			self.circle_xy = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_xy_color,
			)
			self.circle_xy.set_fill(self.circle_xy_color)
			self.circle_xy.set_opacity(0.1)
			self.add(self.circle_xy)

		if self.circle_xz_show:
			self.circle_xz = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_xz_color,
			)
			self.circle_xz.rotate(90 * DEGREES, RIGHT)
			self.circle_xz.set_fill(self.circle_xz_color)
			self.circle_xz.set_opacity(0.1)
			self.add(self.circle_xz)

		if self.circle_yz_show:
			self.circle_yz = Circle(
				radius=SPHERE_RADIUS,
				color=self.circle_yz_color,
			)
			self.circle_yz.rotate(90 * DEGREES, UP)
			self.circle_yz.set_fill(self.circle_yz_color)
			self.circle_yz.set_opacity(0.1)
			self.add(self.circle_yz)

	def init_text(self):
		"""
		for each state, write (with its own color):
			the probabilities
			theta & phi
		"""
		# the qquad is used as a placeholder, since the value changes, and the length of the value changes.
		# Blue qubit is used as base qubit, coord will always be displayed
		# Red one is the fiber one, when entangled, coord when Blue is measured at 0 + angle 
		
		# coord to be edited : 1,3,5,7
		self.tex_tensored = tex("\\ket{\\varphi} = ", "1.0", "\\ket{00} + ", "0.0", "\\ket{01} + ", "0.0", "\\ket{10} + ", "0.0", "\\ket{11}")
		self.tex_tensored.set_color(ORANGE)
		self.tex_tensored.move_to(Z_AXIS * 3)

		self.tex_matrix = tex("Transformation : ", "\\\\", "\\\\", "\\qquad \\begin{bmatrix} 1.000 & 0.000 \\\ 0.000 & 1.000 \end{bmatrix}")
		self.tex_matrix.set_color(WHITE)
		self.tex_matrix.move_to(- Z_AXIS * 2 + Y_AXIS * 3)

		self.tex_zero_vec   = tex("\\ket{q_0} = ", "\\qquad \\qquad 1", " \\\\ ", "\\qquad 0")
		self.tex_zero_vec.set_color(BLUE)
		self.tex_zero_vec.move_to(Z_AXIS * 2 - Y_AXIS * 4)

		self.tex_zero_theta = tex("\\theta = ", "0.000")
		self.tex_zero_theta.set_color(BLUE)
		self.tex_zero_theta.move_to(Z_AXIS * 1 - Y_AXIS * 4)

		self.tex_zero_phi   = tex("\\phi = ", "0.000")
		self.tex_zero_phi.set_color(BLUE)
		self.tex_zero_phi.move_to(Z_AXIS * 0.5 - Y_AXIS * 4)


		self.tex_one_vec    = tex("\\ket{q_1} = ", "\\qquad \\qquad 0", " \\\\ ", "\\qquad 1")
		self.tex_one_vec.set_color(RED)
		self.tex_one_vec.move_to(Z_AXIS * 2 + Y_AXIS * 3.5)

		self.tex_one_theta  = tex("\\theta = ", "180.0")
		self.tex_one_theta.set_color(RED)
		self.tex_one_theta.move_to(Z_AXIS * 1 + Y_AXIS * 4)

		self.tex_one_phi    = tex("\\phi = ", "0.000")
		self.tex_one_phi.set_color(RED)
		self.tex_one_phi.move_to(Z_AXIS * 0.5 + Y_AXIS * 4)

		self.tex_etg= tex("Entangled : ", "No")
		self.tex_etg.set_color(WHITE)
		self.tex_etg.move_to(- Z_AXIS * 2 - Y_AXIS * 5.5)

		self.add(
			self.tex_tensored,
			self.tex_matrix,

			self.tex_zero_vec,
			self.tex_zero_theta,
			self.tex_zero_phi,

			self.tex_one_vec,
			self.tex_one_theta,
			self.tex_one_phi,

			self.tex_etg,
		)

		# the initial values are only used to make enough space for later values
		self.play(
			*self.update_tex_transforms(self.zero, self.one, self.matrix, self.tensored),
			run_time=0.1
		)

	def update_tex_transforms(self, new_zero, new_one, new_matrix, new_tensored):
		zero_state = new_zero.get_vector()
		zero_angles = vector_to_angles(zero_state)
		one_state = new_one.get_vector()
		one_angles = vector_to_angles(one_state)
		print(new_matrix)
		ent_str = "No"
		if ((self.tensored[0]!=0 )& (self.tensored[3] !=0) & (self.tensored[1] == 0) & (self.tensored[2] == 0)) or ((self.tensored[1]!=0) & (self.tensored[2] !=0) & (self.tensored[0] == 0) & (self.tensored[3] == 0)):
			ent_str = "Yes"
		else:
			pass

		return(
			transform(self.tex_matrix[3], matrix_to_tex_string(new_matrix)),

			transform(self.tex_zero_vec[1],   complex_to_str(zero_state[0])),
			transform(self.tex_zero_vec[3],   complex_to_str(zero_state[1])),
			transform(self.tex_zero_theta[1], angle_to_str(zero_angles[0]) ),
			transform(self.tex_zero_phi[1],   angle_to_str(zero_angles[1]) ),

			transform(self.tex_one_vec[1],    complex_to_str(one_state[0]) ),
			transform(self.tex_one_vec[3],    complex_to_str(one_state[1]) ),
			transform(self.tex_one_theta[1],  angle_to_str(one_angles[0])  ),
			transform(self.tex_one_phi[1],    angle_to_str(one_angles[1])  ),

			transform(self.tex_tensored[1], complex_to_str(round(new_tensored[0],2))),
			transform(self.tex_tensored[3], complex_to_str(round(new_tensored[1],2))),
			transform(self.tex_tensored[5], complex_to_str(round(new_tensored[2],2))),
		    transform(self.tex_tensored[7], complex_to_str(round(new_tensored[3],2))),

			transform(self.tex_etg[1], ent_str),
		)

	def init_states(self):
		# Init tensored different to e1 and compute q0 and q1
		self.old_zero = self.zero = State(1, 0, r=2)
		self.old_one  = self.one  = State(1, 0, r=2)
		self.zero.set_color(BLUE)
		self.one.set_color(RED)
		self.old_matrix = self.matrix = Matrix()
		self.tensored = np.array([1,0,0,0])
		self.add(self.zero, self.one, self.matrix)

	def apply_operator(self, operator, verbose=True):
		# preparing the rotation animation
		vgA = VGroup(self.old_zero.line)
		vgB = VGroup(self.old_one.line)
		if self.rotate_sphere:
			vgA.add(self.sphere)
			vgB.add(self.sphere)

		if self.rotate_circles:
			if self.circle_xy_show:
				vgA.add(self.circle_xy)
				vgB.add(self.circle_xy)
			if self.circle_xz_show:
				vgA.add(self.circle_xz)
				vgB.add(self.circle_xz)
			if self.circle_yz_show:
				vgA.add(self.circle_yz)
				vgB.add(self.circle_yz)

		# Updating the tensored product
		vector_result = operator @ self.tensored #matrix vector product
		self.tensored = vector_result
		
		####################################################################################################
		# We need to compute the state of each qubit depending on the tensored state. Handling each cases  #
		# whatever they are entanged or not. If not, we're attributing values to qubits according to the   #
		# rule previously mentionned. If they are, we're basing our computation on 2 qubit bloch Sphere    #
		# paper and the code associated to it.                                                             #
		####################################################################################################

		#Checking if qubits are entangled
		if ((self.tensored[0]!=0 )& (self.tensored[3] !=0) & (self.tensored[1] == 0) & (self.tensored[2] == 0)) or ((self.tensored[1]!=0) & (self.tensored[2] !=0) & (self.tensored[0] == 0) & (self.tensored[3] == 0)):
			if self.tensored[0] < 1e-5:
				print("Intrication 1")
				new_zero = State(self.tensored[1], self.tensored[2], r = 2)
				new_zero.set_color(BLUE)
				alpha1 = 0
				beta1 = 1
			else:
				print("Intrication 2")
				new_zero = State(self.tensored[0], self.tensored[3], r = 2)
				new_zero.set_color(BLUE)
				alpha1 = 1
				beta1 = 0

		else:
			print("Pas intriqué")
			mp.dps=15; mp.pretty=True; pi2=2*np.pi; s2=1/np.sqrt(2)
			## MPMath Ranges:  asin(x)=[-pi/2, pi/2]; acos(x)=[pi, 0]; atan(x)=(-pi/2, pi/2); 
			### atan2(y,x) returns angle (-pi, pi] for correct x,y quadrant ###
			### For Quaternion Calculus:
			### Quaternion imaginary units i, j, k
			i=quaternionic.array([0,1,0,0])
			j=quaternionic.array([0,0,1,0])
			k=quaternionic.array([0,0,0,1])

			### You may Edit: the Real and Imaginary Parts for Four Amplitudes
			alpha_real=self.tensored[0].real; alpha_imag=self.tensored[0].imag; 
			beta_real=self.tensored[1].real; beta_imag=self.tensored[1].imag; 
			gamma_real=self.tensored[2].real; gamma_imag=self.tensored[2].imag; 
			delta_real=self.tensored[3].real; delta_imag=self.tensored[3].imag;
			
			### Do Not Edit 
			aq=quaternionic.array([alpha_real, 0, 0, alpha_imag])
			bq=quaternionic.array([beta_real, 0, 0, beta_imag])
			gq=quaternionic.array([gamma_real, 0, 0, gamma_imag])
			dq=quaternionic.array([delta_real, 0, 0, delta_imag])

			### Or you can use the following for Rotation Gates on 2-qubit Quantum Circuit
			### First give rotation angles eta, omega and nu
			### Then, uncomment the aq, bq, gq and dq for the circuit of your choice

			######@@@@@@@@@@@
			### FOR A QUANTUM CIRCUIT WITH ROTATION OPERATORS ###
			### Rx,y(eta) on Control Qubit
			### C-Rx,y(omega)
			### Rx,y,z(nu) on qubit-A or qubit-B
			### Input: Rotation Operator Angles R(eta) x I, C-R(omega), and R(nu) x I or I x R(nu) 
			eta=60  ##degrees, to induce initial superposition  
			omega=70 #controlled-Rotation Angle in degrees
			nu=0  #single-qubit rotation 

			### Degree to Radian.  DO NOT EDIT.
			eta=eta/180; omega=omega/180; nu=nu/180
			eta=eta*np.pi/2; omega=omega*np.pi/2; nu=nu*np.pi/2  ##2eta = rot angle, etc.   
			## to trigonometric values.  Do not edit
			seta=np.sin(eta); ceta=np.cos(eta)
			somega=np.sin(omega); comega=np.cos(omega)
			snu=np.sin(nu); cnu=np.cos(nu)

			###***** CALCULATE: x0, x1, bt, b, t (Basis=Qubit-A) using Quaternion*****###
			abar=np.conjugate(aq);  bbar=np.conjugate(bq)
			x0_coor=2*(np.power(np.absolute(aq),2)+np.power(np.absolute(bq),2))-1  #xo coordinate of qubit-A state
			x1bt=2*(abar*gq+bbar*dq)+2*(aq*dq-bq*gq)*j ## x1 + bt
			x1_coor=x1bt.real  #x1 coordinate of qubit-A state
			bt=x1bt.imag
			b_norm=np.linalg.norm(bt)  #b_norm=|b|
			b_coor=b_norm  #b-coordinate qubit-A sphere with (x1,b,x0) coordinate
			x1_coor = float(x1_coor) #for some reason, x1_coor is an array which needs to be converted into a float to avoid error by atan2 function
			cz_coor=bt[2]  #z-coord of concurrent circle on inner sphere
			if b_norm !=0:
				if bt[2] < 0:  #use only the Northern Hemisphere for t-sphere: chi > PI/2
					t=-bt/b_norm  #t=array representing pure imaginary unit quaternion
					b_coor=-b_norm
					cz_coor=-bt[2]
				else:       #bt[2]>=0
					t=bt/b_norm
					
			###*** Base-qubit(Qubit-A)Angles: Needed for Fiber-qubit(qubit-B); not for Entanglement, qubit-A ***###
			thetaA=float(mp.acos(x0_coor)) % pi2 ; 
			if (thetaA % np.pi) ==0 :
				phiA=0 #thetaA = 0 or pi: by assumption because undefined
			else:
				#phiA=float(mp.acos(x1_coor/mp.sin(thetaA))) % pi2
				phiA=float(mp.atan2(b_coor, x1_coor)) % pi2
			new_zero = angles_to_vector(thetaA,phiA) 
			alpha1 = (self.tensored[0] + self.tensored[2])/(new_zero[0]+new_zero[1])
			beta1 = (self.tensored[1] + self.tensored[3])/(new_zero[0]+new_zero[1])
			new_zero = State(*angles_to_vector(thetaA,phiA))
			new_zero.set_color(BLUE)
		
		rmA = RotationMatrix(np.array([[1,0],[0,1]]))
		rmB = RotationMatrix(np.array([[1,0],[0,1]]))
		print(rmA.axis)
		print(rmA.theta)
		new_matrix = np.round(operator, 3)
		new_tensored = self.tensored

		new_one = State(alpha1,beta1, r = 2)
		new_one.set_color(RED)


		####################################################################################################
		# To animate the animation of the effect of the operator on the pair of qubits, we need to 		   #
		# determine axis and angle of rotation. Axis is detemined thanks to cross-product since the vector #
		# are ploted with polar coordinate (so R^3). Angles are obtained with  det properties 			   #
		####################################################################################################


		## We need to find the angle between the previous vector stored in self and the new_zero//new_one
		old_zero = (self.zero)._get_cartesian()
		old_one = (self.one)._get_cartesian()
		new_zero_vec = (new_zero)._get_cartesian()
		new_one_vec = (new_one)._get_cartesian()
		
		## Using the cross product to compute a vector who will be orthogonal to both old and new State
		rotA_axis = np.cross(old_zero, new_zero_vec)
		rotB_axis = np.cross(old_one, new_one_vec)


		if np.linalg.norm(rotA_axis,2) != 0:

			## We make sure to normalize them

			rotA_axis = rotA_axis/np.linalg.norm(rotA_axis,2)

			## Obtening angles, we can notice it computes theta/2, that's why there's a 2*

			angleA = 2*np.math.atan2(np.linalg.det([self.zero.get_vector(),new_zero.get_vector()]),np.dot(self.zero.get_vector(),new_zero.get_vector()))

			## Gates of rotation around a normalized axis with rotation theta

			UA = Rv(rotA_axis[0],rotA_axis[1],rotA_axis[2], angleA* 180/np.pi * DEGREES)

			## Decomposing rotations around vector v into rotations around X, Y and Z axis 

			rmA = RotationMatrix(UA)
			if (np.round(RD(rotA_axis[0], rotA_axis[1], rotA_axis[2], angleA*180/np.pi * DEGREES) @ old_zero,3) != np.round(new_zero_vec,3)).any(): ## Handling case angle differ from a factor -1
				rmA.theta = -1 * rmA.theta
			else:
				pass
		else:
			rmA = RotationMatrix(np.array([[1,0], [0,1]]))
			
		if np.linalg.norm(rotB_axis,2) != 0:
			rotB_axis = rotB_axis/np.linalg.norm(rotB_axis,2)
			angleB = 2*np.math.atan2(np.linalg.det([self.one.get_vector(),new_one.get_vector()]),np.dot(self.one.get_vector(),new_one.get_vector()))
			UB = Rv(rotB_axis[0],rotB_axis[1],rotB_axis[2], angleB* 180/np.pi * DEGREES)
			rmB = RotationMatrix(UB)
			if (np.round(RD(rotB_axis[0], rotB_axis[1], rotB_axis[2], angleB*180/np.pi * DEGREES) @ old_one,3) != np.round(new_one_vec,3)).any(): ## Handling case angle differ from a factor -1
				rmB.theta = -1 * rmB.theta
			else:
				pass
		else:
			rmB = RotationMatrix(np.array([[1,0], [0,1]]))

	
		'''
		## For debug
		print("affichage de UA ")
		print(RD(rotA_axis[0], rotA_axis[1], rotA_axis[2], angleA*180/np.pi * DEGREES) @ old_zero)
		print(new_zero_vec)
		print("Affichage de UB")
		print(RD(rotB_axis[0], rotB_axis[1], rotB_axis[2], angleB*180/np.pi * DEGREES) @ old_one)
		print(new_one_vec)
		'''
	
		
		
		self.matrix = operator
		if verbose:
			print(f"rotating around axis: {rmA.axis} by {rmA.theta / DEGREES} degrees")
			print(f"rotating around axis : {rmB.axis} by {rmB.theta / DEGREES} degrees")

		
		'''
		if (new_matrix == np.array([[0.707, 0.707], [0.707,-0.707]])).all():
			new_matrix = Matrix([["\frac{1}{\sqrt{2}}", "\frac{1}{\sqrt{2}}"], ["\frac{1}{\sqrt{2}}", "- \frac{1}{\sqrt{2}}"]])
		'''
		#Matrix library not working since this manim version is too old, so had to create a specific class as a placeholder but can't print latex formulas

		self.play(
			Rotate(
				vgA,
				angle=rmA.theta,
				axis=rmA.axis
			),
			Rotate(
				vgB,
				angle=rmB.theta,
				axis=rmB.axis
			),
			*self.update_tex_transforms(new_zero, new_one, new_matrix, new_tensored),
			run_time=self.rotate_time
		)

		self.zero = new_zero
		self.one  = new_one

class BlochSphereWalk(BlochSphere):
	CONFIG = {
		"show_intro": False,

		"traj_max_length": 0, # 0 is infinite
	}
	def construct(self):
		if self.show_intro:
			self.present_introduction()
		self.init_camera()
		self.init_axes()
		self.init_sphere()
		self.init_states()
		self.init_text()
		self.wait(self.pre_operators_wait_time)

		self.traj_zero = self.add_trajectory(self.old_zero, TEAL_C)

		self.update_theta_and_phi()
		
		self.wait(self.final_wait_time)

	def update_theta_and_phi(self):
		theta = 0
		phi   = 0

		# update theta and phi by calling self.update_state

	def update_state(self, theta, phi, wait=None):
		print(f"theta={theta} ; phi={phi}")
		new_zero = State(*angles_to_vector(theta, phi), r=2)
		new_zero.set_color(BLUE)

		traj = self.traj_zero
		new_point = new_zero.line.get_end()
		if get_norm(new_point - traj.points[-1]) > 0.01:
			traj.add_smooth_curve_to(new_point)
		traj.set_points(traj.points[-self.traj_max_length:])
		
		self.play(
			Transform(self.old_zero, new_zero),
			*self.update_tex_transforms(new_zero, self.one),
			run_time=0.045, # 1/90 = 0.011111111111111112
		)

		if wait:
			self.wait(wait)

		return new_zero

	def add_trajectory(self, state, color=None):
		traj = VMobject()
		traj.set_color(color or state.get_color())
		traj.state = state

		traj.start_new_path(state.line.get_end())
		traj.set_stroke(state.get_color(), 1, opacity=0.75)

		self.add(traj)
		return traj
