import numpy as np

# 設定邊界閥值
import user_set

# 推估頭部姿態
class headpose:
	
	# sum為累計角度
	# coount累計次數 
	# past紀錄上個frame角度
	yaw_sum = np.zeros(3)
	yaw_count = np.zeros(3)
	pitch_sum = np.zeros(3)
	pitch_count = np.zeros(3)
	roll_sum = np.zeros(3)
	roll_count = np.zeros(3)

	deg_past = np.zeros(3)
	
	# 使用單一frame的三軸角度資訊推估頭部姿態
	def headpose_status(yaw, pitch, roll):

		up_down = ''
		left_right = ''
		tilt = ''

		if(yaw > user_set.H_R):			# 大於 +閥值	
			left_right = 'right'
		elif(yaw < user_set.H_L):		# 小於 -閥值
			left_right = 'left'
		else:							# -閥值 ~ +閥值
			left_right = 'normal'

		if(pitch > user_set.H_D):		# 大於 +閥值	
			up_down = 'down'
		elif(pitch < user_set.H_U):		# 小於 -閥值
			up_down = 'up'
		else:							# -閥值 ~ +閥值
			up_down = 'normal'

		if(roll > user_set.T_L):		# 大於 +閥值	
			tilt = 'left'
		elif(roll < user_set.T_R):		# 小於 -閥值
			tilt = 'right'
		else:							# -閥值 ~ +閥值
			tilt = 'normal'

		return left_right, up_down, tilt

	# 統計多個frames的角度資訊
	def headpose_series(yaw, pitch, roll):

		# 當前角度與上一角度差距小於8度時進行統計
		if(abs(yaw - headpose.deg_past[0])<8):
			#yaw
			if(yaw>user_set.H_R):											# 大於 +閥值
				headpose.yaw_sum[0] = headpose.yaw_sum[0] + yaw 			# 累計 +角度
				headpose.yaw_count[0] = headpose.yaw_count[0] +1 			# 累計 +次數

			elif(yaw<user_set.H_L):											# 小於 -閥值
				headpose.yaw_sum[2] = headpose.yaw_sum[2] + yaw 			# 累計 -角度
				headpose.yaw_count[2] = headpose.yaw_count[2] +1 			# 累計 -次數

			else:
				headpose.yaw_count[1] = headpose.yaw_count[1] +1 			# 累計 正常次數 

			headpose.deg_past[0] = yaw 										# 更新紀錄值


		# 當前角度與上一角度差距小於8度時進行統計
		if(abs(pitch - headpose.deg_past[1])<8):	
			#pitch
			if(pitch>user_set.H_D):											# 大於 +閥值
				headpose.pitch_sum[0] = headpose.pitch_sum[0] + pitch		# 累計 +角度
				headpose.pitch_count[0] = headpose.pitch_count[0] +1 		# 累計 +次數

			elif(pitch<user_set.H_U):										# 小於 -閥值
				headpose.pitch_sum[2] = headpose.pitch_sum[2] + pitch		# 累計 -角度
				headpose.pitch_count[2] = headpose.pitch_count[2] +1 		# 累計 -次數

			else:
				headpose.pitch_count[1] = headpose.pitch_count[1] +1 		# 累計 正常次數 

			headpose.deg_past[1] = pitch 									# 更新紀錄值


		# 當前角度與上一角度差距小於8度時進行統計
		if(abs(roll - headpose.deg_past[2])<8):
			#roll
			if(roll>user_set.T_L):											# 大於 +閥值
				headpose.roll_sum[0] = headpose.roll_sum[0] + roll			# 累計 +角度
				headpose.roll_count[0] = headpose.roll_count[0] +1 			# 累計 +次數

			elif(roll<user_set.T_R):										# 小於 -閥值
				headpose.roll_sum[2] = headpose.roll_sum[2] + roll			# 累計 -角度
				headpose.roll_count[2] = headpose.roll_count[2] +1			# 累計 -次數

			else:
				headpose.roll_count[1] = headpose.roll_count[1] +1 			# 累計 正常次數 

			headpose.deg_past[2] = roll 									# 更新紀錄值

	# 依據統計資訊推估頭部姿態
	def headpose_output():

		# 頭部姿態
		left_right = ''
		up_down = ''
		tilt = ''

		# 正常次數最多 
		if(headpose.yaw_count[1] > headpose.yaw_count[0] and headpose.yaw_count[1] > headpose.yaw_count[2]):
			left_right = "normal"
		# +次數最多，+角度平均>10
		elif(headpose.yaw_count[0] > headpose.yaw_count[2] and headpose.yaw_sum[0]/headpose.yaw_count[0] >10):
			left_right = "right"
		# -次數最多，+角度平均<-10
		elif(headpose.yaw_count[0] < headpose.yaw_count[2] and headpose.yaw_sum[2]/headpose.yaw_count[2] < -10):
			left_right = "left"
		else:
			left_right = "normal"

		# 正常次數最多 	
		if(headpose.pitch_count[1] > headpose.pitch_count[0] and headpose.pitch_count[1] > headpose.pitch_count[2]):
			up_down = "normal"
		# +次數最多，+角度平均>10
		elif(headpose.pitch_count[0] > headpose.pitch_count[2] and headpose.pitch_sum[0]/headpose.pitch_count[0] >10):
			up_down = "down"
		# -次數最多，+角度平均<-10
		elif(headpose.pitch_count[0] < headpose.pitch_count[2] and headpose.pitch_sum[2]/headpose.pitch_count[2] < -10):
			up_down = "up"
		else:
			up_down = "normal"

		# 正常次數最多
		if(headpose.roll_count[1] > headpose.roll_count[0] and headpose.roll_count[1] > headpose.roll_count[2]):
			tilt = "normal"
		# +次數最多，+角度平均>10
		elif(headpose.roll_count[0] > headpose.roll_count[2] and headpose.roll_sum[0]/headpose.roll_count[0] >10):
			tilt = "left"
		# -次數最多，+角度平均<-10
		elif(headpose.roll_count[0] < headpose.roll_count[2] and headpose.roll_sum[2]/headpose.roll_count[2] < -10):
			tilt = "right"
		else:
			tilt = "normal"

		return left_right, up_down, tilt

	# 清除統計資訊
	def clear():
		headpose.yaw_sum = np.zeros(3)
		headpose.yaw_count = np.zeros(3)
		headpose.pitch_sum = np.zeros(3)
		headpose.pitch_count = np.zeros(3)
		headpose.roll_sum = np.zeros(3)
		headpose.roll_count = np.zeros(3)

# 計算綜合危險值
def dis_head(dis_status, lr, ud, ti):

	score = 0 		# 總和
	score_d = 0 	# 駕駛行為危險值
	score_lr = 0 	# left_right危險值
	score_ud = 0 	# up_down危險值
	score_ti = 0 	# tilt危險值
	
	if(dis_status != 'safe'):
		score_d = 40

	# 頭部姿態危險值，計數不同有不同危險值
	if(lr != 'normal' and (headpose.yaw_count[0] > 7 or headpose.yaw_count[2] > 7)):
		score_lr = 40
	elif(lr != 'normal' and (headpose.yaw_count[0] > 3 or headpose.yaw_count[2] > 3)):
		score_lr = 30
	elif(lr != 'normal' and (headpose.yaw_count[0] > 1 or headpose.yaw_count[2] > 1)):
		score_lr = 10

	# 頭部姿態危險值，計數不同有不同危險值
	if(ud != 'normal' and (headpose.pitch_count[0] > 7 or headpose.pitch_count[2] > 7)):
		score_ud = 40
	elif(ud != 'normal' and (headpose.pitch_count[0] > 3 or headpose.pitch_count[2] > 3)):
		score_ud = 30
	elif(ud != 'normal' and (headpose.pitch_count[0] > 1 or headpose.pitch_count[2] > 1)):
		score_ud = 10

	# 頭部姿態危險值，計數不同有不同危險值
	if(ti != 'normal' and (headpose.roll_count[0] > 7 or headpose.roll_count[2] > 7)):
		score_ti = 30
	elif(ti != 'normal' and (headpose.roll_count[0] > 3 or headpose.roll_count[2] > 3)):
		score_ti = 20
	elif(ti != 'normal' and (headpose.roll_count[0] > 1 or headpose.roll_count[2] > 1)):
		score_ti = 10

	# 加總危險值
	score = score_d + np.sqrt(score_lr*score_lr + score_ud*score_ud) + score_ti
	
	#print(score, score_d, score_lr, score_ud, score_ti)
	
	return score