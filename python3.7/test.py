# configurable_training_manager.py
import asyncio
import json
import gzip
import time
from typing import Dict, Any, Optional, List
from audio_manager import DialogSession
import protocol
from openai import AzureOpenAI


class ConfigurableTrainingManager:
    def __init__(self, ws_config: Dict[str, Any], config: Dict[str, Any] = None):
        self.session = DialogSession(ws_config)
        self.conversation_state = "greeting"
        self.current_topic = None
        self.conversation_history = []
        self.round_count = 0
        self.max_rounds = 6
        self.douban_initialized = False
        self.role_init_attempts = 0  # 角色初始化尝试次数(弃用）
        self.max_init_attempts = 3  # 最大初始化尝试次数（弃用）
        self.training_completed = False  # 新增：标记培训是否完成
        self.summary_sent = False  # 新增：标记总结是否已发送

        # 配置参数
        default_config = {
            "use_gpt4o": True,
            "enable_gpt4o_logging": True,
            "enable_douban_logging": True,
            "max_rounds": 6,
            "response_length_limit": 200,
            "temperature": 0.85,
            "enable_round_control": True,
            "douban_role_init": True,
            "auto_disconnect": False,  # 新增：是否自动断开连接
        }

        self.config = {**default_config, **(config or {})}
        self.max_rounds = self.config["max_rounds"]

        self.print_config()

        # 初始化Azure OpenAI客户端
        if self.config["use_gpt4o"] or self.config["douban_role_init"]:
            self.azure_client = AzureOpenAI(
                api_key="5fea49cd1d9b404598ed9d2259738486",
                azure_endpoint="https://ll274349293.openai.azure.com/",
                api_version="2024-08-01-preview",
                timeout=30.0,
            )
            print("Azure GPT-4o 客户端初始化成功")
        else:
            self.azure_client = None
            print("使用豆包原生回复模式")

        # 培训讲师的System Prompt
        self.system_prompt = """
你现在的角色是一个资深企业培训师，你的学生来培训的目标是学习一门课程，课程的名字是：《出海》。你的任务是和你的学员进行互动问答，任务要求如下：
1.和学员进行总轮数为6轮的互动问答，最终目的是让学员结合案例学习到课程《出海》中的知识内容。
2.如果学员回复的答案不好，要有耐心，并引导学员往正确的学习方向回答。
3.学员通过案例要回答的问题为：企业如何正确地审视自身，并根据自身的条件和外部环境制定出海战略？
4.你的第一次回复应该是提问问题。当认为学员学习完成了本次课程的内容，基于学员对问题的回答做一个总结和概括学员的学习情况。
5.每次提问和回答都尽量结合案例，让用户结合本次材料的案例来回答和分析。

案例故事：中能科技
[2021年初，中能科技的董事长李总在企业战略会上指出，国内新能源电池行业已经高度饱和，国内竞争异常激烈，利润空间大幅缩小。通过市场分析，欧洲新能源产业正在快速崛起，并且欧洲多国对新能源电池行业表现出强烈的政策支持。
  董事会决定由战略部门、研发部门、法务部门共同成立海外事业小组，专门负责欧洲市场的拓展工作。海外事业小组花费半年时间进行市场调研、风险分析和政策研究，最终决定选择在德国建立首个示范工厂和研发中心。
  他们首先聘请了国际咨询公司，明确了当地政策环境、竞争格局以及准入门槛。同时，迅速开展商务法律谈判，严格锁定了土地成本和基础设施配套费用，避免了不可控的成本上涨风险。
中能科技积极推行"属地化"管理，聘请了许多当地资深技术和管理人才，组建了国际化团队，以融合当地文化和市场。李总还积极地参加当地新能源产业论坛，与当地政府和企业领袖建立起良好的个人关系和信任基础。
  示范项目的成功，引起了德国及周边国家的关注，逐渐中能科技在欧洲市场声誉鹊起。同时企业发现，欧洲政府特别看重智能化和绿色能源方案，于是中能科技进一步加大研发投入，将企业原有的新能源技术与人工智能结合起来，打造出了"智慧储能解决方案"。该解决方案在当地大受欢迎，进一步提升了企业竞争优势和品牌影响力。
  在成功运营的基础上，中能科技又牵头带动国内上下游相关企业一起出海欧洲，构建了完整的新能源产业链集群，与当地政府签订了长期战略合作协议，形成了"政企合作"的良性发展格局。]

请根据当前是第几轮对话来调整你的提问深度和引导方式。每次回复请控制在150-200字以内，保持培训讲师的专业性和亲和力。
"""

    def print_config(self):
        """打印当前配置"""
        print("\n" + "=" * 50)
        print("参数设置")
        print("=" * 50)
        print(f"回复模型: {'GPT-4o' if self.config['use_gpt4o'] else '豆包'}")
        print(f"豆包角色初始化: {'开启' if self.config['douban_role_init'] else '关闭'}")
        print(f"最大轮数: {self.config['max_rounds']}")
        print(f"回复长度限制: {self.config['response_length_limit']}字")
        print(f"GPT-4o Temperature: {self.config['temperature']}")
        print(f"GPT-4o日志: {'开启' if self.config['enable_gpt4o_logging'] else '关闭'}")
        print(f"豆包日志: {'开启' if self.config['enable_douban_logging'] else '关闭'}")
        print(f"轮数控制: {'开启' if self.config['enable_round_control'] else '关闭'}")
        print(f"自动断开连接: {'开启' if self.config['auto_disconnect'] else '关闭'}")
        print("=" * 50 + "\n")

    async def initialize_douban_role(self):
        """使用GPT-4o生成豆包角色初始化内容"""
        try:
            print("开始初始化豆包角色...")

            role_init_prompt = """
请生成一段明确的角色设定指令，用来告诉豆包AI它现在要扮演的角色。

具体要求：
1. 直接告诉它："你现在是一位资深企业培训师"
2. 明确课程名称：《企业出海》培训课程
3. 说明培训目标：通过中能科技案例学习企业出海战略制定
4. 要求它进行6轮互动问答
5. 让它在理解后回复："明白了，我现在是企业培训师，负责《企业出海》课程培训"
6. 语气要像给AI下达明确指令，使用"你现在要..."、"你的任务是..."等
7. 控制在150字以内，确保指令清晰明确

请直接给出这段角色设定指令。
"""

            messages = [
                {"role": "system", "content": "你是一个AI指令生成助手，专门生成清晰明确的角色设定指令。"},
                {"role": "user", "content": role_init_prompt}
            ]

            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,  # 降低温度确保指令更准确
                max_tokens=200,
            )

            role_init_text = response.choices[0].message.content.strip()
            print(f"角色初始化指令生成完成")
            print(f"指令内容: {role_init_text}")

            return role_init_text

        except Exception as e:
            print(f"生成豆包角色初始化失败: {e}")
            # 提供更明确的备用初始化文本
            return """你现在要扮演一位资深企业培训师，负责《企业出海》培训课程。你的任务是基于中能科技进军欧洲的案例，与学员进行6轮互动问答，引导他们学习企业如何制定出海战略。请用培训师的专业语气回复，每次150字左右。请回复"明白了，我现在是企业培训师，负责《企业出海》课程培训"确认你的角色。"""

    async def start_configurable_session(self):
        """启动可配置的培训会话"""
        try:
            await self.session.client.connect()

            # 根据配置选择响应处理器
            if self.config["use_gpt4o"]:
                self.session.handle_server_response = self.gpt4o_response_handler
                print("使用 GPT-4o 响应处理器")

                # 使用GPT-4o生成开场白
                try:
                    print("开始生成GPT-4o开场白...")
                    opening_response = await self.generate_gpt4o_response("开始培训")
                    print(f"开场白生成成功")
                    await self.send_training_content(opening_response)
                except Exception as e:
                    print(f"GPT-4o开场白生成失败，使用默认开场白: {e}")
                    default_opening = "大家好！欢迎参加《企业出海》培训课程。让我们从中能科技的案例开始，请问您认为企业在制定出海战略时，首先应该考虑哪些因素？"
                    await self.send_training_content(default_opening)
            else:
                self.session.handle_server_response = self.douban_response_handler
                print("使用豆包原生响应处理器")

                # 豆包角色初始化
                if self.config["douban_role_init"] and self.azure_client:
                    await self.perform_role_initialization()

            # 启动音频处理
            asyncio.create_task(self.session.process_microphone_input())
            asyncio.create_task(self.session.receive_loop())

            while self.session.is_running:
                # 修改：移除自动断开逻辑，改为手动控制
                if (self.config["enable_round_control"] and
                        self.round_count >= self.max_rounds and
                        not self.training_completed):

                    print(f"已完成 {self.max_rounds} 轮对话，准备发送培训总结...")
                    self.training_completed = True

                    # 等待当前回复完成
                    await asyncio.sleep(3)

                    # 发送培训总结
                    if not self.summary_sent:
                        await self.send_training_summary()
                        self.summary_sent = True

                # 检查是否需要自动断开（可配置）
                if (self.config["auto_disconnect"] and
                        self.training_completed and
                        self.summary_sent):
                    print("培训已完成，5秒后自动断开连接...")
                    await asyncio.sleep(5)
                    break

                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"培训会话错误: {e}")
        finally:
            # 只有在配置为自动断开时才关闭连接
            if self.config["auto_disconnect"]:
                try:
                    print("正在关闭会话连接...")
                    await self.session.client.close()
                except Exception as e:
                    print(f"关闭连接时出错: {e}")
            else:
                print("培训已完成，连接保持开启。可继续对话或手动结束。")

    async def send_training_summary(self):
        """发送培训总结"""
        try:
            if self.config["use_gpt4o"] and self.azure_client:
                try:
                    summary = await self.generate_training_summary()
                    await self.send_training_content(summary)
                    print(f"培训总结内容: {summary}")
                    print("GPT-4o培训总结发送完成")
                except Exception as e:
                    print(f"生成GPT-4o总结失败: {e}")
                    # 使用备用总结
                    await self.send_fallback_summary()
            else:
                try:
                    douban_summary = f"请作为培训讲师对学员在{self.max_rounds}轮《企业出海》培训中的表现进行总结。评价学员对中能科技案例的理解和对企业出海战略制定的掌握情况。给出鼓励性的结束语，控制在200字以内。"
                    print(f"豆包总结指令: {douban_summary}")
                    await self.send_training_content(douban_summary)
                    print("豆包培训总结发送完成")
                except Exception as e:
                    print(f"发送豆包总结失败: {e}")
                    await self.send_fallback_summary()
        except Exception as e:
            print(f"发送培训总结失败: {e}")

    async def send_fallback_summary(self):
        """发送备用总结"""
        fallback_summary = """感谢大家参加今天的《企业出海》培训课程！
        
    通过刚才的6轮互动交流，我们一起深入分析了中能科技成功进军欧洲市场的案例。希望大家能够从中领悟到企业制定出海战略的关键要素：深入的市场调研、准确的自我定位、属地化的经营策略，以及持续的创新能力。
    
    祝愿大家在今后的工作中能够运用这些知识，为企业的国际化发展贡献力量！如果还有任何问题，欢迎继续交流讨论。"""

        print(f"备用培训总结内容: {fallback_summary}")  # 新增这行
        await self.send_training_content(fallback_summary)
        print("备用培训总结发送完成")

    async def perform_role_initialization(self):
        """执行角色初始化流程"""
        try:
            role_init_text = await self.initialize_douban_role()
            print("发送豆包角色初始化指令...")
            await self.send_training_content(role_init_text)

            # 等待豆包处理角色设定
            print("等待豆包确认角色...")
            await asyncio.sleep(3)

            # 设置初始化超时
            self.role_init_start_time = time.time()

        except Exception as e:
            print(f"豆包角色初始化失败: {e}")
            # 如果初始化失败，直接标记为已初始化，开始正常培训
            self.douban_initialized = True

    def gpt4o_response_handler(self, response: Dict[str, Any]):
        """GPT-4o模式的响应处理器"""
        if response == {}:
            return

        if self.config["enable_gpt4o_logging"]:
            pass

        if response['message_type'] == 'SERVER_ACK' and isinstance(response.get('payload_msg'), bytes):
            self.session.audio_queue.put(response['payload_msg'])
            if self.config["enable_gpt4o_logging"]:
                # print("音频数据已加入播放队列")
                pass

        elif response['message_type'] == 'SERVER_FULL_RESPONSE':
            if response.get('event') == 451:  # ASR结果
                if self.config["enable_gpt4o_logging"]:
                    print("收到ASR识别结果")

                user_text = self.extract_asr_text(response)
                if user_text:
                    print(f"ASR识别成功: {user_text}")
                    # 检查是否是结束指令
                    if self.is_end_command(user_text):
                        asyncio.create_task(self.handle_manual_end())
                    else:
                        asyncio.create_task(self.process_user_input_with_gpt4o(user_text))
                else:
                    if self.config["enable_gpt4o_logging"]:
                        print("ASR识别为空或临时结果")

            elif response.get('event') == 450:  # 清空音频缓存
                if self.config["enable_gpt4o_logging"]:
                    print("清空音频缓存")
                while not self.session.audio_queue.empty():
                    try:
                        self.session.audio_queue.get_nowait()
                    except:
                        continue

            elif response.get('event') == 550:  # 拦截豆包模型回复
                if self.config["enable_douban_logging"]:
                    try:
                        douban_content = response.get('payload_msg', {}).get('content', '')
                        print(f"拦截豆包回复: {douban_content}")
                    except:
                        print("拦截豆包回复（无法解析内容）")
                return

        elif response['message_type'] in ['SERVER_ERROR', 'SERVER_FULL_RESPONSE']:
            if response.get('event') in [152, 153]:
                print("会话结束信号")
                self.session.is_session_finished = True

    def douban_response_handler(self, response: Dict[str, Any]):
        """豆包原生模式的响应处理器"""
        if response == {}:
            return

        # 处理ASR结果和轮数统计
        if response['message_type'] == 'SERVER_FULL_RESPONSE':
            if response.get('event') == 451:  # ASR结果
                user_text = self.extract_asr_text(response)
                if user_text:
                    # 检查是否是结束指令
                    if self.is_end_command(user_text):
                        asyncio.create_task(self.handle_manual_end())
                    else:
                        self.handle_user_input_in_douban_mode(user_text)

            elif response.get('event') == 550:  # 豆包回复
                if self.config["enable_douban_logging"]:
                    try:
                        douban_content = response.get('payload_msg', {}).get('content', '')
                        self.handle_douban_response(douban_content)
                    except Exception as e:
                        print(f"豆包回复: [无法解析内容] {e}")

            elif response.get('event') == 559:  # 豆包回复结束
                self.handle_douban_response_end()

        # 使用原始的默认处理逻辑
        try:
            if response['message_type'] == 'SERVER_ACK' and isinstance(response.get('payload_msg'), bytes):
                self.session.audio_queue.put(response['payload_msg'])
            elif response['message_type'] in ['SERVER_ERROR', 'SERVER_FULL_RESPONSE']:
                if response.get('event') in [152, 153]:
                    print("会话结束信号")
                    self.session.is_session_finished = True
        except Exception as e:
            print(f"豆包响应处理警告: {e}")

    def is_end_command(self, user_text: str) -> bool:
        """检查是否是结束指令"""
        end_commands = [
            "结束", "结束培训", "培训结束", "结束会话", "结束对话",
            "bye", "goodbye", "再见", "结束了", "停止", "退出"
        ]

        user_text_lower = user_text.lower().strip()
        return any(cmd in user_text_lower for cmd in end_commands)

    async def handle_manual_end(self):
        """处理手动结束指令"""
        print("收到结束指令，准备结束会话...")

        # 发送结束确认
        end_message = "好的，培训会话即将结束。感谢您的参与！再见！"
        await self.send_training_content(end_message)

        # 等待播放完成
        await asyncio.sleep(3)

        # 关闭连接
        try:
            print("正在关闭会话连接...")
            await self.session.client.close()
            self.session.is_running = False
        except Exception as e:
            print(f"关闭连接时出错: {e}")

    def handle_user_input_in_douban_mode(self, user_text: str):
        """在豆包模式下处理用户输入"""
        if not self.douban_initialized:
            # 角色初始化阶段（仅在启用初始化时）
            print(f"角色初始化阶段 - 用户说: {user_text}")

            # 增加初始化尝试次数
            self.role_init_attempts += 1

            # 降低超时时间，更快进入培训模式
            timeout_duration = 15  # 从30秒降低到15秒

            # 超时或尝试次数过多时强制开始正常培训
            if (hasattr(self, 'role_init_start_time') and
                time.time() - self.role_init_start_time > timeout_duration) or \
                    self.role_init_attempts >= self.max_init_attempts:
                print("角色初始化超时或尝试次数过多，强制开始培训")
                self.douban_initialized = True
                asyncio.create_task(self.send_force_start_message())

        else:
            # 正常对话阶段（关闭初始化时直接进入此阶段）
            if self.config["enable_round_control"]:
                self.round_count += 1
                print(f"第{self.round_count}轮 - 用户说: {user_text}")

                self.conversation_history.append({
                    "role": "user",
                    "content": f"第{self.round_count}轮学员回答: {user_text}"
                })

    def handle_douban_response(self, douban_content: str):
        """处理豆包的回复内容"""
        if not douban_content:
            return

        if not self.douban_initialized:
            print(f"角色初始化回复: {douban_content}")

            # 更宽松的角色确认检测 - 包含更多可能的确认表达
            role_keywords = [
                "明白", "培训师", "企业", "出海", "课程", "中能科技", "讲师",
                "做企业培训", "培训", "教", "负责", "学习", "案例"
            ]

            # 检查是否包含角色相关的关键词
            if any(keyword in douban_content for keyword in role_keywords):
                print("检测到角色相关关键词，豆包可能已理解角色")
                self.douban_initialized = True
                # 发送第一个培训问题
                asyncio.create_task(self.send_first_training_question())

        else:
            print(f"豆包回复: {douban_content}")

            if not hasattr(self, '_current_douban_response'):
                self._current_douban_response = ""
            self._current_douban_response += douban_content

    def handle_douban_response_end(self):
        """处理豆包回复结束"""
        if (hasattr(self, '_current_douban_response') and
                self._current_douban_response and
                self.douban_initialized and
                self.round_count > 0):
            self.conversation_history.append({
                "role": "assistant",
                "content": f"第{self.round_count}轮讲师回复: {self._current_douban_response}"
            })
            delattr(self, '_current_douban_response')

    async def send_force_start_message(self):
        """强制开始培训消息"""
        try:
            start_message = "现在开始《企业出海》培训课程。我们将通过中能科技进军欧洲市场的案例来学习企业出海战略。请问，您认为中能科技在决定出海时，首先分析了哪些关键因素？"
            await self.send_training_content(start_message)
            print("发送强制开始培训消息")
        except Exception as e:
            print(f"发送强制开始消息失败: {e}")

    async def send_first_training_question(self):
        """发送第一个培训问题"""
        try:
            first_question = "很好！现在让我们开始《企业出海》课程的学习。基于中能科技的案例，请您分析一下：企业在制定出海战略时，应该首先考虑哪些内外部因素？"
            await self.send_training_content(first_question)
            print("发送第一个培训问题")
        except Exception as e:
            print(f"发送第一个培训问题失败: {e}")

    async def process_user_input_with_gpt4o(self, user_text: str):
        """使用Azure GPT-4o处理用户输入"""
        self.round_count += 1
        print(f"第{self.round_count}轮 - 用户说: {user_text}")
        print(f"开始处理用户输入...")

        self.conversation_history.append({
            "role": "user",
            "content": f"第{self.round_count}轮学员回答: {user_text}"
        })

        try:
            print("正在调用GPT-4o...")
            start_time = time.time()
            response_text = await self.generate_gpt4o_response(user_text)
            generation_time = time.time() - start_time
            print(f"GPT-4o生成完成，耗时: {generation_time:.2f}秒")

            start_tts = time.time()
            await self.send_training_content(response_text)
            tts_time = time.time() - start_tts
            print(f"TTS发送完成，耗时: {tts_time:.2f}秒")
            print(f"总响应时间: {(generation_time + tts_time):.2f}秒")

        except Exception as e:
            print(f"GPT-4o生成回复失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")

            fallback_response = "非常好的思考！让我们继续深入探讨这个话题。"
            print(f"使用备用回复: {fallback_response}")
            await self.send_training_content(fallback_response)

    async def generate_gpt4o_response(self, user_input: str) -> str:
        """使用Azure GPT-4o生成培训讲师回复"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]

            recent_history = self.conversation_history[-10:] if len(
                self.conversation_history) > 10 else self.conversation_history
            messages.extend(recent_history)

            if user_input == "开始培训":
                current_prompt = """
这是培训的开始，请作为资深企业培训师，结合中能科技的案例，给出一个专业的开场白。

要求：
1. 欢迎学员参加《企业出海》培训课程
2. 简要介绍课程目标：学习企业如何制定出海战略
3. 提出第一个引导性问题，让学员思考中能科技案例中的关键决策
4. 语气要专业但亲和，控制在150-200字

请直接给出开场白，不要说"好的"、"当然"等多余的话。
"""
            else:
                current_prompt = f"""
当前是第{self.round_count}轮对话（总共{self.max_rounds}轮）。
学员刚才说: "{user_input}"

请根据以下情况生成合适的培训讲师回复：
1. 如果这是第1轮，请提出开场问题
2. 如果是中间轮次（2-{self.max_rounds - 1}轮），请针对学员回答给出评价和进一步引导
3. 如果是第{self.max_rounds}轮，请准备总结和评价学员的整体表现

请结合中能科技的案例，引导学员思考企业如何制定出海战略。
回复请控制在{self.config['response_length_limit']}字以内，保持培训讲师的专业性和亲和力。
请直接给出回复内容，不要说"好的"、"当然"等多余的话。
"""

            messages.append({"role": "user", "content": current_prompt})

            if self.config["enable_gpt4o_logging"]:
                print(f"GPT-4o 请求信息:")
                print(f"轮数: {self.round_count}/{self.max_rounds}")
                print(f"用户输入: {user_input}")
                print(f"Temperature: {self.config['temperature']}")

            api_start = time.time()
            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=300,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )
            api_time = time.time() - api_start

            generated_text = response.choices[0].message.content.strip()

            if self.config["enable_gpt4o_logging"]:
                print(f"GPT-4o API调用耗时: {api_time:.2f}秒")
                print(f"响应状态: 成功")
                print(f"响应内容: {generated_text}")
                print(f"响应长度: {len(generated_text)}字")
                try:
                    print(f"Token使用: {response.usage.total_tokens}")
                except:
                    print("Token使用: 未知")

            if len(generated_text) < 20:
                print(f"GPT-4o回复过短，可能有问题: '{generated_text}'")

            if user_input != "开始培训":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"第{self.round_count}轮讲师回复: {generated_text}"
                })

            return generated_text

        except Exception as e:
            if self.config["enable_gpt4o_logging"]:
                print(f"GPT-4o 调用失败:")
                print(f"错误信息: {e}")
                import traceback
                print(f"详细traceback: {traceback.format_exc()}")
            return "让我们继续深入讨论这个重要话题。请分享您的具体想法。"

    async def generate_training_summary(self) -> str:
        """生成培训总结"""
        try:
            summary_prompt = f"""
基于以上{self.max_rounds}轮对话，请作为培训讲师对学员的学习情况进行总结评价：

1. 总结学员对"企业如何制定出海战略"这个核心问题的理解程度
2. 评价学员在案例分析方面的表现
3. 指出学员的进步和需要继续加强的地方
4. 给出鼓励性的结束语

请保持培训讲师的风格，语言要专业但亲和，总结控制在250字以内。
"""

            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": summary_prompt})

            if self.config["enable_gpt4o_logging"]:
                print(f"生成培训总结...")

            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                top_p=0.95,
            )

            summary = response.choices[0].message.content.strip()

            if self.config["enable_gpt4o_logging"]:
                print(f"培训总结生成完成: {len(summary)}字")

            return f"培训总结：{summary}\n\n感谢大家参与今天的《企业出海》培训课程！"

        except Exception as e:
            print(f"生成培训总结失败: {e}")
            return "通过今天的深入交流，我看到了大家对企业出海战略的深入思考。希望大家能够将今天学到的知识应用到实际工作中。感谢参与！"

    async def send_training_content(self, content: str):
        """发送培训内容进行TTS"""
        try:
            print(f"准备发送TTS内容")
            chunks = self.split_text_for_tts(content)
            print(f"文本分段完成，共{len(chunks)}段")

            for i, chunk in enumerate(chunks):
                is_start = (i == 0)
                is_end = (i == len(chunks) - 1)
                print(f"发送第{i + 1}/{len(chunks)}段")
                await self.send_chat_tts_chunk(chunk, is_start, is_end)

            print("所有TTS内容发送完成")

        except Exception as e:
            print(f"发送培训内容失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")

    def split_text_for_tts(self, text: str, max_length: int = 120) -> List[str]:
        """为TTS优化的文本分段"""
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = text.replace('。', '。|').replace('！', '！|').replace('？', '？|').split('|')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk]

    async def send_chat_tts_chunk(self, content: str, start: bool, end: bool):
        """发送ChatTTSText事件块"""
        try:
            chat_tts_request = bytearray(
                protocol.generate_header(
                    message_type=protocol.CLIENT_FULL_REQUEST,
                    message_type_specific_flags=protocol.MSG_WITH_EVENT,
                    serial_method=protocol.JSON,
                    compression_type=protocol.GZIP
                ))

            chat_tts_request.extend(int(500).to_bytes(4, 'big'))
            chat_tts_request.extend((len(self.session.session_id)).to_bytes(4, 'big'))
            chat_tts_request.extend(str.encode(self.session.session_id))

            payload = {
                "start": start,
                "content": content,
                "end": end
            }

            payload_bytes = str.encode(json.dumps(payload))
            payload_bytes = gzip.compress(payload_bytes)
            chat_tts_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
            chat_tts_request.extend(payload_bytes)

            await self.session.client.ws.send(chat_tts_request)

        except Exception as e:
            print(f"发送TTS块失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")

    def extract_asr_text(self, response: Dict[str, Any]) -> Optional[str]:
        """提取ASR识别文本"""
        try:
            payload_msg = response.get('payload_msg', {})
            results = payload_msg.get('results', [])
            if not results:
                return None

            for result in results:
                is_interim = result.get('is_interim', True)
                text = result.get('text', '')

                if not is_interim and text.strip():
                    return text.strip()

            return None
        except Exception as e:
            print(f"提取ASR文本失败: {e}")
            return None


# 使用示例和配置
async def main():
    import config

    # 设置为企业培训讲师
    config.start_session_req["dialog"]["bot_name"] = "企业培训讲师"

    # 配置参数
    training_config = {
        "use_gpt4o": False,  # 改为 True 使用GPT-4o，False 使用豆包原生
        "douban_role_init": False,  # 是否使用GPT-4o初始化豆包角色
        "enable_gpt4o_logging": True,  # 是否显示GPT-4o详细日志
        "enable_douban_logging": True,  # 是否显示豆包回复日志
        "max_rounds": 6,  # 最大对话轮数
        "response_length_limit": 180,  # 回复长度限制
        "temperature": 0.85,  # GPT-4o创造性（0-1）
        "enable_round_control": True,  # 是否启用轮数控制
        "auto_disconnect": False,  # 是否自动断开连接（新增）
    }

    print("开始启动培训会话...")
    print("配置说明：")
    print("   - 豆包原生模式")
    print("   - 6轮对话后自动发送总结")
    print("   - 连接保持开启，支持继续对话")
    print("   - 说'结束'或'再见'可手动结束会话")
    print("-" * 50)

    # 创建培训管理器
    training_manager = ConfigurableTrainingManager(
        ws_config=config.ws_connect_config,
        config=training_config
    )

    # 启动培训会话
    await training_manager.start_configurable_session()


if __name__ == "__main__":
    asyncio.run(main())
