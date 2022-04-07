FROM registry.baidubce.com/paddlepaddle/paddle:2.2.2-gpu-cuda11.2-cudnn8
# apt换源
RUN sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list &&\
        sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN apt-get update && apt-get install openssh-server -y
# pip换源
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
# 添加SSH公钥
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config &&\
        echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config &&\
        echo "AuthorizedKeysFile  .ssh/authorized_keys" >> /etc/ssh/sshd_config &&\
        /etc/init.d/ssh restart &&\
        mkdir -p ~/.ssh &&\
        echo $SSH_PUBKEY > ~/.ssh/authorized_keys


ENTRYPOINT ["/usr/sbin/sshd", "-D"]
